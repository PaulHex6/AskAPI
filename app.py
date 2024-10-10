import streamlit as st
import requests
import json
import sqlite3
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Initialize the client variable
client = None

# Set page config
st.set_page_config(page_title="AskAPI", page_icon="⚡")

# Load .env file and get the OpenAI API key from it
load_dotenv()
api_key_env = os.getenv("OPENAI_API_KEY", "")

# Initialize session state variables if not already set
if 'api_list' not in st.session_state:
    st.session_state['api_list'] = []  # Initialize with an empty list
if 'selected_api' not in st.session_state:
    st.session_state['selected_api'] = None
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = api_key_env or ''
if 'debug_output' not in st.session_state:
    st.session_state['debug_output'] = ''  # Debug output storage

# Sidebar: Show intro text always
st.sidebar.title("AskAPI")
st.sidebar.markdown("**AskAPI** lets you interact with any API using natural language.")

# Sidebar: Show API key input only if not already provided
if not st.session_state['api_key']:
    st.sidebar.markdown("Please enter your OpenAI API key:")
    st.session_state['api_key'] = st.sidebar.text_input(
        "API Key",
        type="password",
        value='',
        placeholder="sk-..."
    )

# Function to add debug information to the debug output box
def add_debug_info(info):
    if 'debug_output' not in st.session_state:
        st.session_state['debug_output'] = ''
    # Check if info is a dictionary (API response) and format it
    if isinstance(info, dict):
        formatted_info = json.dumps(info, indent=2)
        st.session_state['debug_output'] += f"{formatted_info}\n"
    else:
        st.session_state['debug_output'] += f"{info}\n"

# Helper Functions

def fetch_documentation_from_url(url):
    """Fetch API documentation from a URL and extract plain text."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n')
    return text

def query_gpt(messages):
    global client
    """Send messages to the AIMLAPI GPT model and return the response."""
    response = client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        max_tokens=10000
    )
    return response.choices[0].message.content.strip()

def build_knowledge_base(doc_text):
    """Use GPT to generate a structured knowledge base from the provided documentation text."""
    combined_content = (
        "You are an AI assistant who translates API documentation into a knowledge base. Your reply is always in JSON format only, no other comments.\n\n"
        f"Analyze the following API documentation and extract the base URL, request methods, parameters, and authentication methods in JSON format:\n\n{doc_text}"
    )

    messages = [
        {"role": "user", "content": combined_content}
    ]

    knowledge_base_str = query_gpt(messages)

    # Debugging: Print the raw GPT response
    add_debug_info(f"Raw GPT Response: {knowledge_base_str}")

    # Extract JSON content from code block
    def extract_json_from_code_block(text):
        code_block_pattern = r'```(?:json)?\n(.*?)```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()

    knowledge_base_str = extract_json_from_code_block(knowledge_base_str)

    try:
        knowledge_base = json.loads(knowledge_base_str)
    except json.JSONDecodeError as e:
        add_debug_info(f"Error parsing JSON from GPT response: {e}")
        knowledge_base = {}

    return knowledge_base

def save_api_to_db(api_url, knowledge_base):
    """Save the API URL and knowledge base to the SQLite database."""
    conn = sqlite3.connect('apis.db')
    c = conn.cursor()
    c.execute("INSERT INTO apis (url, knowledge_base) VALUES (?, ?)", (api_url, json.dumps(knowledge_base)))
    conn.commit()
    conn.close()

def get_api_list():
    """Retrieve the list of APIs from the database."""
    conn = sqlite3.connect('apis.db')
    c = conn.cursor()
    c.execute("SELECT id, url FROM apis")
    result = c.fetchall()
    conn.close()
    return result

def get_knowledge_base(api_id):
    """Retrieve the knowledge base of a specific API from the database."""
    conn = sqlite3.connect('apis.db')
    c = conn.cursor()
    c.execute("SELECT knowledge_base FROM apis WHERE id = ?", (api_id,))
    result = c.fetchone()
    conn.close()
    return json.loads(result[0]) if result else None

def extract_parameters(user_query, knowledge_base):
    """Use GPT to extract relevant API parameters based on the user's query."""
    add_debug_info(f"User Query: {user_query}")
    add_debug_info(f"Knowledge Base: {json.dumps(knowledge_base, indent=2)}")

    combined_content = (
        "You are an AI assistant who translates user query to API parameters.\n\n"
        f"Given the following user query: '{user_query}', and API knowledge base: <knowledge>{json.dumps(knowledge_base, indent=2)}</knowledge>, extract the necessary API parameters in JSON format."
    )

    messages = [
        {"role": "user", "content": combined_content}
    ]

    params_text = query_gpt(messages)

    # Debugging: Print the raw GPT response before cleaning
    add_debug_info(f"Raw GPT Response: {params_text}")

    # Extract JSON content from code block
    def extract_json_from_code_block(text):
        code_block_pattern = r'```(?:json)?\n(.*?)```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()

    params_text = extract_json_from_code_block(params_text)

    # Debugging: Print the cleaned GPT response after removing non-JSON parts
    add_debug_info(f"Cleaned GPT Response: {params_text}")

    try:
        # Parse the cleaned JSON response
        params = json.loads(params_text)
        add_debug_info(f"Extracted Parameters: {json.dumps(params, indent=2)}")

        return params

    except json.JSONDecodeError as e:
        # Log the error in the debug output
        add_debug_info(f"Error parsing JSON: {e}")
        return {}

def create_api_call(knowledge_base, params):
    """Create the full API call based on the knowledge base and extracted parameters."""

    # Debugging: Ensure the base_url is available
    base_url = knowledge_base.get('base_url')

    if not base_url:
        add_debug_info("Error: 'base_url' is missing from the knowledge base. Cannot make API call.")
        return {"error": "Missing base_url"}

    method = knowledge_base.get('request_methods', ['GET'])[0].upper()
    headers = knowledge_base.get('headers', {})

    # Use the extracted 'params' as the query parameters in the API call
    query_params = params  # Using the extracted parameters directly
    body = params.get('body', {})

    add_debug_info(f"Making {method} request to {base_url} with params: {query_params} and body: {body}")

    try:
        # Handle GET and POST requests based on the method specified
        if method == 'GET':
            api_url = base_url
            response = requests.get(api_url, headers=headers, params=query_params)
        elif method == 'POST':
            api_url = base_url
            response = requests.post(api_url, headers=headers, json=body)
        else:
            return {"error": "Unsupported request method."}

        # Even if Content-Type is not application/json, try to manually parse the response as JSON
        try:
            return response.json()
        except ValueError:
            # If JSON parsing fails, return the raw text response
            add_debug_info(f"Manual JSON parsing: Response is not in JSON format, but content looks like JSON.")
            return {"error": "Response is not in JSON format.", "raw_response": response.text}

    except requests.exceptions.RequestException as e:
        add_debug_info(f"API request failed: {e}")
        return {"error": f"API request failed: {str(e)}"}

def summarize_response(user_query, api_response):
    """Summarize the API response into natural language using GPT, considering the user's initial query."""
    combined_content = (
        "You are an AI assistant that provides clear answers to user queries based on API responses.\n\n"
        f"The user asked: '{user_query}'. Based on the following API response, provide a concise and relevant answer to the user's question:\n\n{json.dumps(api_response, indent=2)}"
    )

    messages = [
        {"role": "user", "content": combined_content}
    ]
    summary = query_gpt(messages)
    return summary

# Main Tabs in the UI
def main():
    global client  # Use the global client variable
    st.title("⚡ AskAPI")

    # Check if API key is set
    if not st.session_state['api_key']:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Initialize AIMLAPI Client (use your own API key and endpoint)
    client = OpenAI(
        api_key=st.session_state['api_key'],
        base_url="https://api.aimlapi.com",
    )

    # Tabs for the app
    tabs = ["Talk to API", "Manage APIs", "Search APIs", "Documentation"]
    selected_tab = st.tabs(tabs)

    with selected_tab[0]:
        talk_to_api_tab()
    with selected_tab[1]:
        manage_apis_tab()
    with selected_tab[2]:
        search_apis_tab()
    with selected_tab[3]:
        documentation_tab()

# Function for 'Talk to API' tab
def talk_to_api_tab():
    st.header("Talk to API")

    # Initialize session state variables if not already set
    if 'api_response' not in st.session_state:
        st.session_state['api_response'] = None
    if 'debug_output' not in st.session_state:
        st.session_state['debug_output'] = ''
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ''

    # Show available APIs
    api_list = get_api_list()
    if api_list:
        api_selection = st.selectbox("Select an API", api_list, format_func=lambda x: x[1])
        st.session_state['selected_api'] = api_selection[0]
    else:
        st.info("No APIs available. Please add one in the 'Manage APIs' tab.")

    # Prompt to ask a query
    user_query = st.text_input("Enter your query:", value=st.session_state['user_query'])
    submit_query = st.button("Submit Query")

    if submit_query:
        # Update user query in session state
        st.session_state['user_query'] = user_query

        # Clear debug output and previous response when a new query is submitted
        st.session_state['debug_output'] = ''
        st.session_state['api_response'] = None

        if not user_query or not st.session_state['selected_api']:
            st.warning("Please select an API and enter a query.")
        else:
            knowledge_base = get_knowledge_base(st.session_state['selected_api'])
            if knowledge_base:
                params = extract_parameters(user_query, knowledge_base)
                api_response = create_api_call(knowledge_base, params)
                st.session_state['api_response'] = api_response  # Store in session state

                if "error" not in api_response:
                    summary = summarize_response(user_query, api_response)
                    st.write("**Response:**")
                    st.write(summary)
                else:
                    st.error(f"Error: {api_response.get('error', 'Unknown error')}")

                # Append Raw API Response to Debug Output
                add_debug_info("Raw API Response:")
                add_debug_info(api_response)

    # Display debug info using expander
    if st.session_state.get('debug_output'):
        with st.expander("Show Debug Output"):
            st.text_area("Debug Output", st.session_state['debug_output'], height=400)

# Function for 'Manage APIs' tab
def manage_apis_tab():
    st.header("Manage APIs")

    # Add new API
    new_api_url = st.text_input("Enter API documentation URL:")
    if st.button("Add API"):
        doc_text = fetch_documentation_from_url(new_api_url)
        knowledge_base = build_knowledge_base(doc_text)
        save_api_to_db(new_api_url, knowledge_base)
        st.success(f"Added API: {new_api_url}")

    # Display available APIs
    api_list = get_api_list()
    if api_list:
        st.dataframe(api_list)

# Function for 'Search APIs' tab
def search_apis_tab():
    st.header("Search APIs")
    search_query = st.text_input("Search for APIs:")
    if search_query:
        api_list = get_api_list()
        st.write(f"Search results for '{search_query}':")
        for api in api_list:
            if search_query.lower() in api[1].lower():
                st.write(api[1])
    else:
        st.info("No APIs added yet.")

# Function for 'Documentation' tab
def documentation_tab():
    st.header("Documentation")
    api_list = get_api_list()
    selected_api = st.selectbox("Select an API for documentation", api_list, format_func=lambda x: x[1])
    if selected_api:
        doc_text = fetch_documentation_from_url(selected_api[1])
        st.text_area("API Documentation", doc_text, height=400)  # Increased height

if __name__ == "__main__":
    main()
