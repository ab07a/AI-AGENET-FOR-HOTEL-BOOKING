import json
import os
from datetime import datetime
from typing import Dict, Any, TypedDict, Annotated, List
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
import operator
import pandas as pd
import prompt
from dm_function import send_message
"17841439819236506"
from sms import send_sms
load_dotenv()

api_key = os.getenv("gemini_api_key")
llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.7,
        google_api_key=api_key,
    )

curr_date = datetime.now()
current_date_for_llm = str(curr_date).split()[0]

df = pd.read_csv("booking_data.csv")
df.set_index("reservation_id",inplace = True)

in_progess = False



class State(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], operator.add]
    intent: str
    current_reservation_id: str
    sender_id: str

def chatBot(state: State):
    print("\n--- Entering chatBot node (for intent classification) ---")
    current_user_input_content = ""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        current_user_input_content = state["messages"][-1].content
    else:
        print("Warning: Last message in chat_history is not a HumanMessage or list is empty. Cannot extract current_user_input_content.")

    formatted_messages = [prompt.hotel_booking_flags_prompt]+state["messages"]+[HumanMessage(content=f"User Input: {current_user_input_content}\n current_booking_progress: {in_progess}")]
    

    response = llm.invoke(formatted_messages)
    print(f"LLM Raw Response from chatBot: {response.content}")
    raw_json_string = response.content
    start_index = raw_json_string.find('{')
    end_index = raw_json_string.rfind('}') + 1

    data = {}
    if start_index != -1 and end_index != 0 and start_index < end_index:
        json_substring = raw_json_string[start_index:end_index]
        try:
            data = json.loads(json_substring)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM: {e}\nProblematic content: {json_substring}")
    else:
        print(f"Warning: No valid JSON object found in LLM response: {raw_json_string}")


    determined_intent = data.get("intent")
    reservation_id = data.get("reservation_id")

    if reservation_id != None and reservation_id != "null":
        return {"messages": [response], "intent": determined_intent, "current_reservation_id": reservation_id}
    return {"messages": [response], "intent": determined_intent}


def book(state: State):
    global in_progess
    in_progess = "TRUE"
    print("\n--- Entering BOOK node ---")
    current_user_input_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_user_input_content = msg.content
            break
    if not current_user_input_content:
        print("Warning: No human input found for booking details.")
        return {"messages": state["messages"] + [AIMessage(content="I couldn't understand your request. Please try again.")]}

    reservation_id_for_llm = "RES"+str(len(df)+1)


    system_message_context = prompt.booking_details_prompt(current_date_for_llm, reservation_id_for_llm)


    formatted_messages = [SystemMessage(content=system_message_context)]+state["messages"]+[HumanMessage(content=f"User input: {current_user_input_content}")]
    response = llm.invoke(formatted_messages)
    print(f"LLM Raw Response from book node: {response.content}")

    try:
        raw_json_string = response.content
        start_index = raw_json_string.find('{')
        end_index = raw_json_string.rfind('}') + 1
        
        if start_index != -1 and end_index != 0 and start_index < end_index:
            json_substring = raw_json_string[start_index:end_index]
            booking_data_output = json.loads(json_substring)
            extracted_booking_details = booking_data_output.get("booking_data", {})
            print(booking_data_output.get("message"))
            send_message(state["sender_id"], booking_data_output.get("message"))
            values_list = list(extracted_booking_details.values())
            df.loc[values_list[0]] = values_list[1:]
            df.to_csv("booking_data.csv")
            print(df)
            return {
                "messages": [response],
                "current_reservation_id": extracted_booking_details.get("reservation_id") 
            }
        else:
            print(f"Warning: No valid JSON object found in LLM response for booking: {raw_json_string}")
            return {"messages": state["messages"] + [AIMessage(content="I had trouble understanding your booking details. Could you please rephrase?")]}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM in book node: {e}\nProblematic content: {response.content}")
        return {"messages": state["messages"] + [AIMessage(content="I encountered an error processing your booking details. Please try again.")]}
def update(state: State):
    global in_progess
    in_progess = "TRUE"
    print("\n--- Entering UPDATE node ---")
    reservation_id = state.get("current_reservation_id")
    data = {}
    if reservation_id in df.index:
        data = df.loc[reservation_id]

    current_user_input_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_user_input_content = msg.content
            break
    if not current_user_input_content:
        print("Warning: No human input found for booking details.")
        return {"messages": state["messages"] + [AIMessage(content="I couldn't understand your request. Please try again.")]}

    data = dict(data)


    if data == {}:
        send_message(state["sender_id"], "Please provide a valid reservation_id or initialize a new booking.")
        return {
                "messages": [AIMessage(content="please provide correct reservation_id")]
            }
    elif data['status'] == "cancelled":
        in_progess = "FALSE"
        send_message(state["sender_id"], "Updation is not possible as this booking is cancelled. Please start a new booking.")
        return {
                "messages": [AIMessage(content="updation is not possible as this booking is cancelled please start a new booking")]
            }
    print("data=======",data,reservation_id)
    update_system_message = prompt.update_details_prompt(current_date_for_llm, reservation_id, data)
    print("data=======",data,reservation_id)
    formatted_messages = [update_system_message]+state["messages"]+[HumanMessage(content=f"User input: {current_user_input_content}")]
    response = llm.invoke(formatted_messages)
    print(f"LLM Raw Response from book node: {response.content}")
    try:
        raw_json_string = response.content
        start_index = raw_json_string.find('{')
        end_index = raw_json_string.rfind('}') + 1
        
        if start_index != -1 and end_index != 0 and start_index < end_index:
            json_substring = raw_json_string[start_index:end_index]
            booking_data_output = json.loads(json_substring)
            extracted_booking_details = booking_data_output.get("data", {})
            print(booking_data_output.get("message"))
            send_message(state["sender_id"], booking_data_output.get("message"))
            values_list = list(extracted_booking_details.values())
            if booking_data_output.get("update_init") and extracted_booking_details.get("status") == "confirmed":
                print("Sending SMS with booking details...")
                send_sms(extracted_booking_details.get("phone_number"), booking_data_output.get("message"))
            if booking_data_output.get("update_init") and extracted_booking_details.get("status") != "cancelled":
                df.loc[reservation_id] = values_list
                df.to_csv("booking_data.csv")
            return {
                "messages": [response]
            }
        else:
            print(f"Warning: No valid JSON object found in LLM response for booking: {raw_json_string}")
            return {"messages": state["messages"] + [AIMessage(content="I had trouble understanding your booking details. Could you please rephrase?")]}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM in book node: {e}\nProblematic content: {response.content}")
        return {"messages": state["messages"] + [AIMessage(content="I encountered an error processing your booking details. Please try again.")]}

def inquire(state: State):
    print("\n--- Entering INQUIRY node ---")
    reservation_id = state.get("current_reservation_id")
    data = {}
    if reservation_id in df.index:
        data = df.loc[reservation_id]

    current_user_input_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_user_input_content = msg.content
            break
    if not current_user_input_content:
        print("Warning: No human input found for booking details.")
        return {"messages": state["messages"] + [AIMessage(content="I couldn't understand your request. Please try again.")]}

    data = dict(data)

    print("data=======",data,reservation_id)
    
    inquire_response_prompt = prompt.inquire_response_prompt(reservation_id, data)
    formatted_messages = [inquire_response_prompt]+state["messages"]+[HumanMessage(content=f"User input: {current_user_input_content}")]
    response = llm.invoke(formatted_messages)
    print(f"LLM Raw Response from book node: {response.content}")
    try:
        raw_json_string = response.content
        start_index = raw_json_string.find('{')
        end_index = raw_json_string.rfind('}') + 1
        
        if start_index != -1 and end_index != 0 and start_index < end_index:
            json_substring = raw_json_string[start_index:end_index]
            booking_data_output = json.loads(json_substring)
            print(booking_data_output.get("message"))
            send_message(state["sender_id"], booking_data_output.get("message"))
            return {
                "messages": [response]
            }
        else:
            print(f"Warning: No valid JSON object found in LLM response for booking: {raw_json_string}")
            return {"messages": state["messages"] + [AIMessage(content="I had trouble understanding your booking details. Could you please rephrase?")]}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM in book node: {e}\nProblematic content: {response.content}")
        return {"messages": state["messages"] + [AIMessage(content="I encountered an error processing your booking details. Please try again.")]}

def qa(state: State):
    print("\n--- Entering qa node ---")
    qa_response_prompt = prompt.qa_response_prompt
    current_user_input_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            current_user_input_content = msg.content
            break
    if not current_user_input_content:
        print("Warning: No human input found for booking details.")
        return {"messages": state["messages"] + [AIMessage(content="I couldn't understand your request. Please try again.")]}
    formatted_messages = [qa_response_prompt]+state["messages"]+[HumanMessage(content=f"User input: {current_user_input_content}")]
    response = llm.invoke(formatted_messages)
    print(f"LLM Raw Response from book node: {response.content}")
    try:
        raw_json_string = response.content
        start_index = raw_json_string.find('{')
        end_index = raw_json_string.rfind('}') + 1
        
        if start_index != -1 and end_index != 0 and start_index < end_index:
            json_substring = raw_json_string[start_index:end_index]
            booking_data_output = json.loads(json_substring)
            print(booking_data_output.get("message"))
            send_message(state["sender_id"], booking_data_output.get("message"))
            return {
                "messages": [response]
            }
        else:
            print(f"Warning: No valid JSON object found in LLM response for booking: {raw_json_string}")
            return {"messages": state["messages"] + [AIMessage(content="I had trouble understanding your booking details. Could you please rephrase?")]}

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM in book node: {e}\nProblematic content: {response.content}")
        return {"messages": state["messages"] + [AIMessage(content="I encountered an error processing your booking details. Please try again.")]}

def select_intent(state: State):
    print(f"Selecting next node based on intent: {state['intent']}")
    return state["intent"]

def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatBot", chatBot)
    builder.add_node("book", book)
    builder.add_node("update", update)
    #builder.add_node("cancel", cancel)
    builder.add_node("inquire", inquire)
    builder.add_node("qa", qa)
    builder.set_entry_point("chatBot")
    builder.add_conditional_edges(
        "chatBot",
        select_intent,
        {
            "BOOK": "book",
            "UPDATE": "update",
            "INQUIRE": "inquire",
            "QA": "qa"
        }
    )
    builder.add_edge("book", END)
    builder.add_edge("update", END)
    builder.add_edge("inquire", END)
    builder.add_edge("qa", END)
    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)