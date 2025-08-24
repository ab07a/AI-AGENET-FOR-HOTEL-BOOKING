from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()
import os

account_sid = os.getenv('account_sid')
auth_token = os.getenv('auth_token')
client = Client(account_sid, auth_token)

def send_sms(to, body):
    message = client.messages.create(
        from_=os.getenv('phone_number'),
        body=body,
        to='+91'+to
    )
    return message.sid