import json
import pandas as pd
from datetime import datetime

def read_telegram_chat(file_path: str) -> pd.DataFrame:
    """
    Read a Telegram chat export JSON file and convert it to a DataFrame.
    
    Args:
        file_path: Path to the Telegram JSON export file
        
    Returns:
        DataFrame with columns: timestamp, sender, message
    """
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract messages
    messages = []
    
    for msg in data.get('messages', []):
        # Skip service messages and empty messages
        if msg.get('type') == 'service' or not msg.get('text'):
            continue
            
        # Extract text from plain text entities or use the text field
        text = ""
        if isinstance(msg['text'], str):
            text = msg['text']
        elif isinstance(msg['text'], list):
            text = ""
        else:
            # Extract text from text_entities if it exists
            for entity in msg.get('text_entities', []):
                if entity.get('type') == 'plain':
                    text += entity.get('text', '')
        
        if text:
            timestamp_str = msg.get('date', '')
            try:
                # Parse ISO format datetime
                timestamp = datetime.fromisoformat(timestamp_str)
            except:
                timestamp = None
                
            sender = msg.get('from', 'Unknown')
            
            messages.append({
                'timestamp': timestamp,
                'sender': sender,
                'message': text
            })
    
    # Create DataFrame
    df = pd.DataFrame(messages)
    
    return df 