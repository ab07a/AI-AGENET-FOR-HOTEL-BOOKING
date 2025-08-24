import json

from flask import Flask
from flask import request
from dm_function import send_message
from llm_model import build_graph
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

graph = build_graph()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/webhook", methods = ["GET", "POST"])
def webhook():
    if request.method == "POST":
        try:
            json_data = json.dumps(request.get_json(), indent=4)
            print(json_data)
            json_data = json.loads(json_data)
            json_data = json_data["entry"][0]["messaging"][0]
            sender_id = json_data["sender"]["id"]
            user_input_str = json_data["message"].get("text", "")
            print(f"Sender ID: {sender_id}")
            if not user_input_str:
                send_message(sender_id, "please provide text input only")
                return "<p>No user input found.</p>"
            if sender_id != os.getenv('my_instagram_id'):
                print(f"User input: {user_input_str}")
                print(f"Sender ID: {sender_id}")
                graph.invoke({"messages": [HumanMessage(content=user_input_str)],"sender_id": sender_id}, config={"configurable": {"thread_id": sender_id}})
        except:
            pass
        return "<p>This is POST Request, Hello Webhook!</p>"
    
    if request.method == "GET":
        hub_mode = request.args.get("hub.mode")
        hub_challenge = request.args.get("hub.challenge")
        hub_verify_token = request.args.get("hub.verify_token")
        if hub_challenge:
            return hub_challenge
        else:
            return "<p>This is GET Request, Hello Webhook!</p>"

if __name__ == "__main__":
    app.run(port=5000)