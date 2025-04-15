import sys
import os
from pathlib import Path
import torch

from core.data_cleaning import clean_data
from core.byte_pair_encoding import train_tokenizer
from core.transformer_model import create_model

def display_instructions():
    """Display instructions for chat file placement"""
    print("\n=== Chat File Placement Instructions ===")
    print("- For WhatsApp chats: Export chat files and place .txt files in the '/data/whatsapp/' folder")
    print("- For Telegram chats: Export chat files and place .json files in the '/data/telegram/' folder")
    print("====================================\n")

def check_directories():
    """Check if necessary directories exist, create them if they don't"""
    base_dir = Path(__file__).parent
    dirs_to_check = [
        base_dir / "data",
        base_dir / "data" / "whatsapp",
        base_dir / "data" / "telegram",
        base_dir / "output",
        base_dir / "output" / "tokenizer"
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    check_directories()
    display_instructions()
    
    print("\n=== Personas.AI Language Model Training Pipeline ===")
    print("This program will extract chat data and train a tokenizer.")
    
    # STEP 1: Extract Chat Data
    print("\n--- STEP 1: Extract Chat Data ---")
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
        whatsapp_username = username
        telegram_username = username
    else:
        # make sure they reference to the same user
        whatsapp_username = input("[WHATSAPP] Enter username to extract messages for: ")
        telegram_username = input("[TELEGRAM] Enter username to extract messages for: ")
    
    # ask for chat types
    print("\nWhich chat types would you like to process?")
    print("1. WhatsApp only")
    print("2. Telegram only")
    print("3. Both WhatsApp and Telegram")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    chat_types = []
    if choice == '1':
        chat_types = ['whatsapp']
        print("\nProcessing WhatsApp chats only.")
    elif choice == '2':
        chat_types = ['telegram']
        print("\nProcessing Telegram chats only.")
    else:
        chat_types = ['whatsapp', 'telegram']
        print("\nProcessing both WhatsApp and Telegram chats.")
    
    messages, total_chars = clean_data(whatsapp_username, telegram_username, chat_types)
    
    print(f"\nExtracted {len(messages)} messages for user '{whatsapp_username}' / '{telegram_username}'")
    print(f"Total characters: {total_chars}")
    
    if len(messages) > 0:
        print(f"\nSuccess! User-specific messages saved to 'output/{whatsapp_username}-{telegram_username}-messages.txt'")
        print("All messages saved to 'output/combined_text.txt' for pre-training")
    else:
        print(f"\nNo messages found for '{whatsapp_username}' / '{telegram_username}'. Please check:")
        print("1. The username is correct (case-insensitive matching is used)")
        print("2. Chat files are in the correct directories")
        print("3. The chat files contain messages from this user")
        
        # exit if no messages found as we can't proceed to training
        print("\nExiting as no user messages were found. Please try again.")
        return
    
    # STEP 2: TRAIN TOKENIZER
    print("\n--- STEP 2: TRAIN TOKENIZER ---")
    
    # ask which data to use for tokenizer training
    print("\nWhich data would you like to use for tokenizer training?")
    print("1. All messages (recommended for pre-training)")
    print("2. User-specific messages only")
    
    data_choice = input("Enter your choice (1-2, default: 1): ").strip()
    
    # use default combined_text.txt for pre-training or user-specific for fine-tuning
    input_file = None  # default path in train_tokenizer
    if data_choice == '2':
        input_file = f"output/{whatsapp_username}-{telegram_username}-messages.txt"
        print(f"\nUsing user-specific messages for tokenizer training.")
    else:
        print("\nUsing all messages for tokenizer training.")
    
    # ask for vocabulary size
    vocab_size = input("Enter vocabulary size (default: 1024): ").strip()
    vocab_size = int(vocab_size) if vocab_size.isdigit() else 1024
    
    # train the tokenizer
    tokenizer = train_tokenizer(input_file=input_file, vocab_size=vocab_size)
    
    # STEP 3: CREATE TRANSFORMER MODEL
    print("\n--- STEP 3: CREATE TRANSFORMER MODEL ---")
    
    # ask for model parameters
    print("\nEnter model parameters (or press Enter for defaults):")
    
    block_size_input = input("Block size (default: 256): ").strip()
    block_size = int(block_size_input) if block_size_input.isdigit() else 256
    
    n_embd_input = input("Embedding dimension (default: 384): ").strip()
    n_embd = int(n_embd_input) if n_embd_input.isdigit() else 384
    
    n_head_input = input("Number of attention heads (default: 6): ").strip()
    n_head = int(n_head_input) if n_head_input.isdigit() else 6
    
    n_layer_input = input("Number of transformer layers (default: 6): ").strip()
    n_layer = int(n_layer_input) if n_layer_input.isdigit() else 6
    
    dropout_input = input("Dropout probability (default: 0.2): ").strip()
    try:
        dropout = float(dropout_input) if dropout_input else 0.2
    except ValueError:
        dropout = 0.2
    
    # create the model
    model, tokenizer = create_model(
        tokenizer=tokenizer,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    
    # example generation
    print("\nGenerating example text:")
    
    # create a simple context
    context = "Hello, how are you"
    encoded_context = tokenizer.encode(context)
    
    # convert to tensor and move to model's device
    context_tensor = torch.tensor([encoded_context], device=model.device)
    
    # generate some tokens
    print(f"Context: '{context}'")
    
    max_new_tokens = 20
    generated = model.generate(context_tensor, max_new_tokens=max_new_tokens)
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print(f"Generated: '{generated_text}'")
    
    print("\n=== Language Model Training Pipeline Complete ===")
    print("You can now use the trained model for generating text.")
    print("Note: The model was trained on all messages for better pre-training.")
    print("For fine-tuning, you can use the user-specific messages.")

if __name__ == "__main__":
    main()