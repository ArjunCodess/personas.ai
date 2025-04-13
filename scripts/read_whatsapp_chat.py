import re
import pandas as pd

def read_whatsapp_chat(file_path: str) -> pd.DataFrame:
    encryption_message = "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them. Tap to learn more."
    media_pattern = "<Media omitted>"
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    edited_message = "<This message was edited>"
    deleted_message = "You deleted this message"
    null_message = "null"
    created_group_message = "created group"
    added_you_to_group_message = "added you"
    tagging_pattern = r'@[\w]+'

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filtered_lines = []
    for line in lines:
        if (
            encryption_message not in line and
            deleted_message not in line and
            null_message != line.split(" ")[-1] and
            media_pattern not in line and
            created_group_message not in line and
            added_you_to_group_message not in line and
            not re.search(email_pattern, line) and
            not re.search(url_pattern, line)
        ):
            line = line.replace(edited_message, "").strip()
            line = re.sub(tagging_pattern, "", line).strip()
            filtered_lines.append(line)

    content = '\n'.join(filtered_lines)
    
    pattern = r'(\d{2}/\d{2}/\d{2,4},\s+\d{1,2}:\d{2}\s*(?:am|pm)?)\s+-\s+([^:]+):\s+(.*?)(?=\n\d{2}/\d{2}/\d{2,4}|$)'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    clean_data = []
    for match in matches:
        timestamp, sender, message = match
        sender = sender.strip()
        message = message.strip()
        clean_data.append((timestamp, sender, message))
    
    df = pd.DataFrame(clean_data, columns=['timestamp', 'sender', 'message'])
    
    timestamps = []
    for ts in df['timestamp']:
        try:
            # try 12-hour format with am/pm first
            if 'am' in ts.lower() or 'pm' in ts.lower():
                timestamp = pd.to_datetime(ts, format='%d/%m/%y, %I:%M %p')
            else:
                # try 24-hour format
                timestamp = pd.to_datetime(ts, format='%d/%m/%Y, %H:%M')
        except:
            try:
                # if year is in 2-digit format without am/pm
                timestamp = pd.to_datetime(ts, format='%d/%m/%y, %H:%M')
            except:
                # last resort fallback
                timestamp = pd.NaT
        timestamps.append(timestamp)
    
    df['timestamp'] = timestamps
        
    return df