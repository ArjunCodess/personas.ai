from pathlib import Path
import os
from scripts.read_whatsapp_chat import read_whatsapp_chat
from scripts.read_telegram_chat import read_telegram_chat

def clean_data(whatsapp_username, telegram_username, chat_types=None):
    """
    Process chat data and extract messages for all users and specific users.
    
    Args:
        whatsapp_username (str): The WhatsApp username to filter messages for
        telegram_username (str): The Telegram username to filter messages for
        chat_types (list): List of chat types to process ('whatsapp', 'telegram', or both)
    
    Returns:
        tuple: (List of messages from the specified users, Total character count for user messages)
    """
    
    if not whatsapp_username and not telegram_username:
        raise ValueError("At least one username parameter is required")
    
    if chat_types is None:
        chat_types = ['whatsapp', 'telegram']
    
    base_dir = Path(__file__).parent.parent
    output_directory = base_dir / "output"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    all_chats = {}
    
    if 'whatsapp' in chat_types:
        whatsapp_directory = base_dir / "data" / "whatsapp"
        if os.path.exists(whatsapp_directory):
            for file in whatsapp_directory.glob('*.txt'):
                file_name = file.stem
                all_chats[f"whatsapp_{file_name}"] = read_whatsapp_chat(file)
    
    if 'telegram' in chat_types:
        telegram_directory = base_dir / "data" / "telegram"
        if os.path.exists(telegram_directory):
            for file in telegram_directory.glob('*.json'):
                file_name = file.stem
                all_chats[f"telegram_{file_name}"] = read_telegram_chat(file)
    
    user_messages = []
    all_messages = []
    
    for chat_id, chat_df in all_chats.items():
        try:
            # collect all messages for pre-training
            all_messages.extend(chat_df['message'].tolist())
            
            # filter messages for specific users (for fine-tuning)
            if chat_id.startswith('whatsapp_') and whatsapp_username:
                user_rows = chat_df[chat_df['sender'].str.lower() == whatsapp_username.lower()]
            elif chat_id.startswith('telegram_') and telegram_username:
                user_rows = chat_df[chat_df['sender'].str.lower() == telegram_username.lower()]
            else:
                continue
                
            if not user_rows.empty:
                user_messages.extend(user_rows['message'].tolist())
        except Exception as e:
            print(f"Error processing chat {chat_id}: {str(e)}")
    
    # save all messages to combined_text.txt for pre-training
    combined_file = output_directory / "combined_text.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write(" ".join(all_messages))
    
    # save user-specific messages to text file
    user_filename = f"{whatsapp_username}-{telegram_username}-messages.txt"
    user_file = output_directory / user_filename
    with open(user_file, "w", encoding="utf-8") as f:
        f.write("\n".join(user_messages))
    
    total_chars = sum(len(msg) for msg in user_messages)
    
    return user_messages, total_chars