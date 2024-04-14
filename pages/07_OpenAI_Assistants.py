from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import json
import openai as client
import streamlit as st
from typing import Type
from bs4 import BeautifulSoup
import requests
import time
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)
def DuckDuckSearchTool(inputs):
    print("DuckDuck search tool inputs:",inputs)
    ddg = DuckDuckGoSearchAPIWrapper()
    print("DuckDuck search start")
    query = inputs["query"]
    results = ddg.run(query)
    # Assuming the results are a JSON object and the first result's URL is accessible via ['results'][0]['url']
    if results and 'results' in results and len(results['results']) > 0:
        url = results['results'][0]['url']  # Access the URL of the first result
        print("DuckDuck url:",url)
    print("DuckDuck results:",results)
    return results


def WikiSearchTool(inputs):
    print("Wiki search tool inputs:",inputs)
    wiki = WikipediaAPIWrapper()
    query = inputs["query"]
    results = wiki.run(query)
    if results and 'results' in results and len(results['results']) > 0:
        url = results['results'][0]['url']  # Access the URL of the first result
        print("Wiki url:",url)
    print("Wiki results:",results)    
    return results

def SaveToTextFileTool(inputs):
    try:
        file_path = f"MyResearch.txt"
        if 'count' not in st.session_state:
            st.session_state['count'] = 0
        mykey = st.session_state['count']
        if "url" in inputs.keys():
            url = inputs["url"]
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            response.raise_for_status()  # Raises an error for bad responses
            text = soup.get_text()
            st.download_button('Download text', text, 'text',key = mykey)
            # with open(file_path, 'w') as f:
            #     f.write(text) 
            #     # st.download_button('Download CSV', f) 
            #     print(f"Saved text from URL to file: {file_path}")
        else:
            text = inputs["text"]
            st.download_button('Download text', text, 'text',key = mykey)
            # with open(file_path, 'w') as f:
            #     f.write(text)
            #     print(f"Saved text from URL to file: {file_path}")
        st.session_state['count'] += 1
        return "Text saved to file."
    except requests.RequestException as e:
        text = inputs["text"]
        st.download_button('Download text', text, 'text',key = mykey)
        st.session_state['count'] += 1
        return f"Failed to load URL: {str(e)}"
    except ValueError:
        text = inputs["text"]
        st.download_button('Download text', text, 'text',key = mykey)
        st.session_state['count'] += 1
        with open(file_path, 'w') as f:
            f.write(text)     
        return "URL did not return a JSON response."  
def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        print(json.loads(function.arguments))
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs    

def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outpus
    )
@st.cache_data
def create_run(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
@st.cache_data
def create_thread(content):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )
     
functions = [
    {
        "type": "function",
        "function": {
            "name": "DuckDuckSearchTool",
            "description": "Use this tool to search for the provided query and provide one url from the searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that user wants to search",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WikiSearchTool",
            "description": "Use this tool to search for the provided query and provide one url from the searching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that user wants to search",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SaveToTextFileTool",
            "description": "Use this tool to save the text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text that user wants to save to the text file.",
                    },
                    "url": {
                        "type": "string",
                        "description": "The url that user wants to enter and extract text to save as the text file.",
                    },
                },
                "required": ["text"],
            },
        },
    },
]



        
    
functions_map = {
    "DuckDuckSearchTool": DuckDuckSearchTool,
    "WikiSearchTool": WikiSearchTool,
    "SaveToTextFileTool": SaveToTextFileTool,
}



count = 0

# Sidebar for API Key and GitHub Repo link
with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Enter your OpenAI API Key")
    # todo: if user change the api key, we need to reinitialize the llm
    # how to save api key in a secure way?
    API_key_check_btn = st.button("Check API KEY")
    if not api_key:
        st.error("Please enter your OpenAI API Key")    
    if st.session_state.get("APIKEY_failed"):  
        st.error("Invalid API Key. Please enter a valid API Key")

    st.write("GitHub Repo:https://github.com/daegyujeong/gpt_pratice_dk/blob/9d8242349586396932d7f030161a55049adfdf12/pages/07_OpenAI_Assistants.py")
    if API_key_check_btn:
        if  api_key:
            llm = None
            # for checking if the API Key is valid
            try:
                llm = ChatOpenAI(
                    api_key=api_key,
                    temperature=0.1,
                    model="gpt-3.5-turbo-1106",
                )
                llm.invoke("Hello!")   
                st.session_state["APIKEY_failed"] = False
                st.success("API Key is valid")
                # You might want to add a simple test call here if the API does not get invoked elsewhere immediately
            except Exception as e:
                st.markdown(f"Failed to initialize ChatOpenAI: {str(e)}")   
                st.session_state["APIKEY_failed"] = True
                st.rerun()       

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_required_action(run_id, thread_id):
    run = get_run(run_id, thread_id)
    return run.required_action

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    last_message = {"role":"", "content":""}
    if len(messages) > 0:
        last_message = {"role":messages[0].role, "content":messages[0].content[0].text.value}
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
        # for debugging
    return last_message
    


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_message(message, role):
    with st.chat_message(role):
        st.markdown(message)

def paint_history():
    for message in st.session_state["messages"]:
        paint_message(
            message["message"],
            message["role"]
        )




st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about stock!
"""
)
if "messages" in st.session_state:
    paint_history()

message = st.chat_input("Ask anything about your file...")
if message:
    if st.session_state["APIKEY_failed"] == False:
        client.api_key = api_key
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""You help users do research from the web.
            Search the query using DuckDuckSearchTool and extract the content from the link.
            Search the query using WikiSearchTool and extract the content from the link.
            if there is link from the content, enter the link and extract the content.
            if there is no link just save text content from the research.
            Save the content as text file.""",
            model="gpt-4-1106-preview",
            tools=functions,
        )    
        print("message :",message)
        send_message(message, "human")
        thread = create_thread(message)
        print(thread.id, assistant.id)
        run = create_run(thread.id, assistant.id)
        print(get_run(run.id, thread.id).status,get_required_action(run.id, thread.id))    
        start_time = time.time()  # Record the start time
        previous_result = "None"
        while True:
            # todo : exception handling
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time > 120:
                print("Timeout reached, stopping the polling.")
                break   
            run_status = get_run(run.id, thread.id).status
               
            if run_status == 'completed':
                print("Run completed successfully.")
                result = get_messages(thread.id)
                print(result)
                if(not(result["content"] == previous_result) and result["role"]== "assistant"):
                    previous_result = result["content"]
                    send_message(result["content"], "assistant")
                break
            elif run_status == 'failed':
                print("Run failed.")
            elif run_status == 'expired':
                print("Run was expired.")
            elif run_status == 'stopped':
                print("Run was stopped.")
            elif run_status == 'requires_action':
                print("Run is requires_action.")  
                run = get_run(run.id, thread.id)
                outputs = []
                for action in run.required_action.submit_tool_outputs.tool_calls:
                    action_id = action.id
                    function = action.function
                    send_message(f"Calling function: {function.name} with arg {function.arguments}", "assistant")                        
                submit_tool_outputs(run.id, thread.id)
                count = count + 1
            elif run_status == 'queued':
                print("Run is queued.")
            elif run_status == 'in_progress':
                print("Run is in_progress.")
            elif run_status == 'cancelling':
                print("Run is cancelling.")
            elif run_status == 'cancelled':
                print("Run is cancelled.")
            print(f"Current run status: {run.status}, waiting for completion...")
            time.sleep(5)  # Wait for 5 seconds before checking again
        result = get_messages(thread.id)
        print(st.session_state["messages"])
        # send_message(result, "assistant")

        # st.rerun()
    else:
        st.error("Invalid API Key. Please enter a valid API Key")
        st.stop()
else:
    st.session_state["messages"] = []