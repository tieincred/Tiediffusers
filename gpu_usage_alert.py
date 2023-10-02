import os
import time
import psutil
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the Slack Webhook URL from environment variable
WEBHOOK_URL = os.getenv('WEBHOOK')
if not WEBHOOK_URL:
    raise ValueError("SLACK_WEBHOOK_URL is not set in the environment variables or .env file.")

# Define the usage thresholds
HIGH_USAGE_THRESHOLD = 1024  # 1 GB in MB
LOW_USAGE_THRESHOLD = 100  # 100 MB
SLEEP_INTERVAL = 60  # 1 minute in seconds

use_alert = False
train_alert = True

def send_slack_notification(message):
    slack_data = {'text': message}
    response = requests.post(
        WEBHOOK_URL, json=slack_data,
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )


def monitor_gpu_usage():
    while True:
        # Get the GPU memory usage. You may need to adjust this depending on your system and GPU.
        # This is a simplistic example, please adjust it according to your specific environment and needs.
        gpu_memory = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
        print(f"Current GPU usage is {gpu_memory}")
        if use_alert and gpu_memory > HIGH_USAGE_THRESHOLD:
            send_slack_notification(f"High GPU usage detected: {gpu_memory} MB")

        if train_alert:
            time.sleep(SLEEP_INTERVAL)  # wait for 1 minute
            gpu_memory_after = psutil.virtual_memory().available / (1024 * 1024)  # Convert to MB
            if gpu_memory_after < LOW_USAGE_THRESHOLD:
                send_slack_notification(f"Low GPU usage detected: {gpu_memory_after} MB for a minute")

        time.sleep(5)  # Check every 5 seconds, adjust as needed


if __name__ == "__main__":
    monitor_gpu_usage()