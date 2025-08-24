import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

user_access_token = os.getenv("access_token")
def send_message(id, message):
    url = f"https://graph.instagram.com/v21.0/me/messages"
    headers = {
        "Authorization": f"Bearer {user_access_token}",
        "Content-Type": "application/json"
    }
    json_body = {
        "recipient": {
            "id": id
        },
        "message": {
            "text": message
        }
    }

    response = requests.post(url, headers=headers, json=json_body)
    data = response.json()
    print(json.dumps(data, indent=4))