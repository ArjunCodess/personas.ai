import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from minbpe import BasicTokenizer

def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    """
    Get the total vocabulary size including special tokens.
    
    Args:
        tokenizer: The BasicTokenizer instance
        
    Returns:
        int: Total vocabulary size
    """
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    return len(vocab) + len(special_tokens)

class Head(nn.Module):
    """ One head of self-attention """
    
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        _, T, _ = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    
    def __init__(self, n_embd: int, num_heads: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out

class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.feed_forward = FeedFoward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    GPT Language Model based on the Transformer architecture.
    """

    def __init__(self, vocab_size: int, block_size: int = 256, n_embd: int = 384, 
                 n_head: int = 6, n_layer: int = 6, dropout: float = 0.2, device: str = None) -> None:
        super().__init__()
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.block_size = block_size
        
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)]
        )
        
        # Output layers
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            input_tokens: Tensor of token indices of shape (batch_size, sequence_length)
            targets: Optional tensor of target token indices of same shape as input_tokens

        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, sequence_length, vocab_size)
            and loss is optional cross-entropy loss if targets are provided
        """
        B, T = input_tokens.shape

        # Input_tokens and targets are both (B,T) tensors of integers
        token_embedding = self.token_embedding_table(input_tokens)  # (B,T,C)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=self.device))  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.final_layer_norm(x)  # (B,T,C)
        logits = self.final_linear_layer(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens given a context.

        Args:
            input_tokens: Starting token indices of shape (batch_size, sequence_length)
            max_new_tokens: Number of new tokens to generate

        Returns:
            Tensor of token indices of shape (batch_size, sequence_length + max_new_tokens)
        """
        # input_tokens is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop input_tokens to the last block_size tokens
            cropped_input = input_tokens[:, -self.block_size:]
            # Get the predictions
            logits, _ = self(cropped_input)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)  # (B, T+1)
        return input_tokens

def create_model(tokenizer=None, tokenizer_path=None, block_size=256, n_embd=384, 
                n_head=6, n_layer=6, dropout=0.2, device=None):
    """
    Create a GPT language model with the specified parameters.
    
    Args:
        tokenizer: Optional BasicTokenizer instance
        tokenizer_path: Path to load the tokenizer from if not provided
        block_size: Maximum sequence length
        n_embd: Embedding dimension
        n_head: Number of attention heads
        n_layer: Number of transformer layers
        dropout: Dropout probability
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        GPTLanguageModel: Initialized model
    """
    # load tokenizer if not provided
    if tokenizer is None and tokenizer_path is None:
        base_dir = Path(__file__).parent.parent
        tokenizer_path = base_dir / "output" / "tokenizer" / "my_tokenizer.model"
    
    if tokenizer is None:
        tokenizer = BasicTokenizer()
        tokenizer.load(model_file=str(tokenizer_path))
    
    # get vocabulary size
    vocab_size = get_vocab_size(tokenizer)
    
    # set default device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # create model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    )
    
    # move model to device
    model = model.to(device)
    
    # print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created model with {num_params/1e6:.2f} M parameters")
    
    return model, tokenizer