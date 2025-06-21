from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('TG_PHONE')

# Paths
MEDIA_DIR = 'photos'
CSV_OUTPUT_PATH = os.path.join('data', 'raw', 'telegram_data.csv')
CHANNELS_FILE_PATH = os.path.join('data', 'raw', 'channels.txt')

# Media download toggle
DOWNLOAD_MEDIA = False  # Set to True if you want to download images

# Ensure photo folder exists
os.makedirs(MEDIA_DIR, exist_ok=True)

# Initialize Telegram client
client = TelegramClient('scraping_session', api_id, api_hash)

# Scrape messages from a single channel
async def scrape_channel(client, channel_username, media_dir):
    rows = []
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        print(f"[...] Scraping {channel_username} ({channel_title})")

        count = 0
        async for message in client.iter_messages(entity, limit=1000):  # Faster for testing
            count += 1
            if count % 50 == 0:
                print(f"  └ Processed {count} messages")

            media_path = None
            if DOWNLOAD_MEDIA and message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)

            rows.append([
                channel_title,
                channel_username,
                message.id,
                message.message,
                message.date,
                media_path
            ])

        print(f"[✓] Done scraping {count} messages from {channel_username}")
        return rows

    except Exception as e:
        print(f"[!] Error scraping {channel_username}: {e}")
        return []

# Main scraping logic
async def main():
    await client.start(phone=phone)

    # Load channel list
    if not os.path.exists(CHANNELS_FILE_PATH):
        print(f"[!] Channel list not found: {CHANNELS_FILE_PATH}")
        return

    with open(CHANNELS_FILE_PATH, 'r', encoding='utf-8') as f:
        channels = [line.strip() for line in f if line.strip()]

    # remove repeatation
    channels = list(dict.fromkeys(channels))
    print(channels, len(channels))
    
    # Check if file exists to add header
    file_exists = os.path.exists(CSV_OUTPUT_PATH)

    for channel in channels:
        channel_data = await scrape_channel(client, channel, MEDIA_DIR)

        if channel_data:
            with open(CSV_OUTPUT_PATH, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
                    file_exists = True
                writer.writerows(channel_data)
                print(f"[✓] Written data for {channel}")
        else:
            print(f"[✗] No data written for {channel}")

# Run it
with client:
    client.loop.run_until_complete(main())
