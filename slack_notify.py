import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_slack_notification():
    # Read the Slack Webhook URL from environment variable
    webhook_url = os.getenv('WEBHOOK')
    if not webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL is not set in the environment variables or .env file.")
    
    slack_data = {'text': "Training Complete!"}
    
    response = requests.post(
        webhook_url, json=slack_data,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )

send_slack_notification()