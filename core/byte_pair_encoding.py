import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent))
from minbpe.basic import BasicTokenizer

def train_tokenizer(input_file=None, output_prefix=None, vocab_size=1024):
    """
    Train a BPE tokenizer on the extracted messages and save it.
    
    Args:
        input_file: Path to the input text file
        output_prefix: Path prefix for saving the tokenizer
        vocab_size: Size of the vocabulary to learn
    """

    if input_file is None:
        base_dir = Path(__file__).parent.parent
        input_file = base_dir / "output" / "combined_text.txt"
    
    if output_prefix is None:
        base_dir = Path(__file__).parent.parent
        output_prefix = base_dir / "output" / "tokenizer" / "my_tokenizer"
        
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    with open(input_file, "r", encoding="utf-8") as f:
        text_sequence = f.read()
    
    tokenizer = BasicTokenizer()
    print(f"Training tokenizer with vocab size {vocab_size}...")
    tokenizer.train(text_sequence, vocab_size=vocab_size)
    
    max_vocab_id = list(tokenizer.vocab.keys())[-1]
    tokenizer.special_tokens = {
        "<|startoftext|>": max_vocab_id + 1,
        "<|separator|>": max_vocab_id + 2,
        "<|endoftext|>": max_vocab_id + 3,
        "<|unk|>": max_vocab_id + 4
    }
    
    tokenizer.save(file_prefix=str(output_prefix))
    print(f"Tokenizer saved to {output_prefix}")
    
    encoded_text = tokenizer.encode(text_sequence)
    print(f"Total tokens in full text: {len(encoded_text)}")
    print(f"Unique tokens used: {len(set(encoded_text))}")
    print(f"Compression ratio: {len(text_sequence) / len(encoded_text):.2f} characters per token")
    
    example_text = "Hello, how are you?"
    encoded = tokenizer.encode(example_text)
    print(f"\nExample encoding:")
    print(f"Text: '{example_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{tokenizer.decode(encoded)}'")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()