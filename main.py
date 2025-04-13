from core.data_cleaning import clean_data
import sys
import os
from pathlib import Path

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
        base_dir / "output"
    ]
    
    for directory in dirs_to_check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    check_directories()
    
    display_instructions()
    
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        # make sure they reference to the same user
        whatsapp_username = input("[WHATSAPP] Enter username to extract messages for: ")
        telegram_username = input("[TELEGRAM] Enter username to extract messages for: ")
    
    # Ask for chat types
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
        print("\nSuccess! Messages saved to 'output/combined_text.txt'")
    else:
        print(f"\nNo messages found for '{whatsapp_username}' / '{telegram_username}'. Please check:")
        print("1. The username is correct (case-insensitive matching is used)")
        print("2. Chat files are in the correct directories")
        print("3. The chat files contain messages from this user")

if __name__ == "__main__":
    main()