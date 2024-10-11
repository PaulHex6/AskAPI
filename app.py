import streamlit as st
import requests
import json
import sqlite3
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import docx  # For reading .docx files

# Initialize the client variable
client = None

# Set page config
st.set_page_config(page_title="AskAPI", page_icon="⚡")

# Load .env file and get the OpenAI API key from it
load_dotenv()
api_key_env = os.getenv("OPENAI_API_KEY", "")

# Initialize session state variables if not already set
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

# Register JSON adapter and converter for SQLite
def adapt_json(data):
    return json.dumps(data)

def convert_json(text):
    return json.loads(text)

sqlite3.register_adapter(dict, adapt_json)
sqlite3.register_converter("JSON", convert_json)

# Function to initialize the database and create the 'apis' table
def initialize_db():
    """Initialize the database and create the 'apis' table if it doesn't exist."""
    conn = sqlite3.connect('apis.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS apis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_name TEXT,
            short_description TEXT,
            base_url TEXT,
            knowledge_base JSON
        )
    ''')
    conn.commit()
    conn.close()

# Function to add debug information to the debug output box
def add_debug_info(info, attempt=None):
    if 'debug_output' not in st.session_state:
        st.session_state['debug_output'] = ''
    prefix = f"Attempt {attempt}: " if attempt else ""
    # Check if info is a dictionary (API response) and format it
    if isinstance(info, dict):
        formatted_info = json.dumps(info, indent=2)
        st.session_state['debug_output'] += f"{prefix}{formatted_info}\n"
    else:
        st.session_state['debug_output'] += f"{prefix}{info}\n"

# Helper Functions

def fetch_documentation_from_url(url):
    """Fetch API documentation from a URL and extract plain text."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator='\n')
    return text

# Send messages to the OpenAI GPT model and return the response.
def query_gpt(messages):
    global client
    with st.spinner('Processing, please wait...'):
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
        "Analyze the following API documentation and extract the following fields in JSON format:\n"
        "- api_name: The name of the API\n"
        "- short_description: A brief description of what the API does, max 60 characters\n"
        "- base_url: The base URL for the API\n"
        "- endpoint: The primary endpoint for the API\n"
        "- request_methods: List of HTTP methods supported\n"
        "- parameters: Parameters accepted by the API\n"
        "- authentication_methods: Authentication methods required\n\n"
        f"API Documentation:\n{doc_text}"
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
        knowledge_base = None
        return None

    # Check if 'base_url' is in knowledge_base
    if 'base_url' not in knowledge_base:
        add_debug_info("Error: 'base_url' is missing from the knowledge base.")
        knowledge_base = None

    return knowledge_base

def save_api_to_db(api_url, knowledge_base):
    """Save the API data and knowledge base to the SQLite database."""
    # Extract 'api_name', 'short_description', 'base_url' from knowledge_base
    api_name = knowledge_base.get('api_name', 'Unknown API')
    short_description = knowledge_base.get('short_description', '')
    base_url = knowledge_base.get('base_url', api_url)  # Fallback to api_url if 'base_url' not found

    conn = sqlite3.connect('apis.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(
        "INSERT INTO apis (api_name, short_description, base_url, knowledge_base) VALUES (?, ?, ?, ?)",
        (api_name, short_description, base_url, json.dumps(knowledge_base))  # Ensure JSON serialization
    )
    conn.commit()
    conn.close()

    # Optional: Add a debug statement to confirm saving
    add_debug_info(f"API '{api_name}' saved successfully with base URL '{base_url}'.")

def get_api_list():
    """Retrieve the list of APIs from the database."""
    conn = sqlite3.connect('apis.db')
    c = conn.cursor()
    c.execute("SELECT id, api_name, short_description, base_url FROM apis")
    result = c.fetchall()
    conn.close()
    return result

def get_knowledge_base(api_id):
    """Retrieve the knowledge base of a specific API from the database."""
    conn = sqlite3.connect('apis.db', detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute("SELECT knowledge_base FROM apis WHERE id = ?", (api_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def extract_parameters(user_query, knowledge_base, error_message=None):
    """Use GPT to extract relevant API parameters based on the user's query and error feedback."""
    add_debug_info(f"User Query: {user_query}")
    add_debug_info(f"Knowledge Base: {json.dumps(knowledge_base, indent=2)}")

    combined_content = (
        "You are an AI assistant who translates user queries into API parameters.\n\n"
        f"Given the following user query: '{user_query}', and API knowledge base: <knowledge>{json.dumps(knowledge_base, indent=2)}</knowledge>.\n"
    )

    if error_message:
        combined_content += f"\nPreviously, the API returned the following error: '{error_message}'. Please adjust the parameters accordingly.\n"

    combined_content += "Extract the necessary API parameters in JSON format. Do not include 'method' or 'url' keys. Provide only the parameters required by the API."

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

def process_parameters(params):
    """Process parameters to ensure they are formatted correctly for the API call."""
    processed_params = {}
    for key, value in params.items():
        if isinstance(value, list):
            # Convert lists to comma-separated strings
            processed_params[key] = ','.join(map(str, value))
        else:
            processed_params[key] = value
    return processed_params

def create_api_call(knowledge_base, params):
    """Create the full API call based on the knowledge base and extracted parameters."""
    # Ensure the base_url and endpoint are available
    base_url = knowledge_base.get('base_url')
    endpoint = knowledge_base.get('endpoint', '')

    if not base_url:
        add_debug_info("Error: 'base_url' is missing from the knowledge base. Cannot make API call.")
        return {"success": False, "error": "Missing base_url"}

    # Construct the full API URL
    api_url = base_url.rstrip('/') + '/' + endpoint.lstrip('/')

    method = knowledge_base.get('request_methods', ['GET'])[0].upper()
    headers = knowledge_base.get('headers', {})

    # Process parameters to ensure correct formatting
    processed_params = process_parameters(params)

    # Include 'body' in the debug message
    add_debug_info(f"Making {method} request to {api_url} with params: {processed_params}")

    try:
        # Handle GET and POST requests based on the method specified
        if method == 'GET':
            response = requests.get(api_url, headers=headers, params=processed_params)
        elif method == 'POST':
            response = requests.post(api_url, headers=headers, json=processed_params)
        else:
            return {"success": False, "error": "Unsupported request method."}

        # Raise an exception for HTTP errors (status codes 4xx or 5xx)
        response.raise_for_status()

        # Parse the response as JSON
        api_response = response.json()
        return {"success": True, "response": api_response}

    except requests.exceptions.RequestException as e:
        error_message = f"API request failed: {str(e)}"
        add_debug_info(error_message)
        return {"success": False, "error": error_message}

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

    # Initialize the database
    initialize_db()

    # Check if API key is set
    if not st.session_state['api_key']:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Initialize OpenAI Client (use your own API key)
    client = OpenAI(
        api_key=st.session_state['api_key'],
        base_url="https://api.aimlapi.com",
    )

    # Tabs for the app
    tabs = st.tabs(["Talk to API", "Manage APIs", "Search APIs", "Documentation"])

    with tabs[0]:
        talk_to_api_tab()
    with tabs[1]:
        manage_apis_tab()
    with tabs[2]:
        search_apis_tab()
    with tabs[3]:
        documentation_tab()

# Function for 'Talk to API' tab
def talk_to_api_tab():
    # Initialize session state variables if not already set
    if 'api_response' not in st.session_state:
        st.session_state['api_response'] = None
    if 'debug_output' not in st.session_state:
        st.session_state['debug_output'] = ''
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ''

    # Fetch the latest API list from the database
    api_list = get_api_list()
    if api_list:
        api_selection = st.selectbox("Select an API", api_list, format_func=lambda api: f"{api[1]}")
        st.session_state['selected_api'] = api_selection[0]
    else:
        st.info("No APIs available. Please add one in the 'Manage APIs' tab.")
        return  # Early exit if no APIs are available

    # Prompt to ask a query
    user_query = st.text_input("Enter your query:", value='', key='query_input')
    submit_query = st.button("Ask API")

    if submit_query:
        # Update user query in session state with the latest query
        st.session_state['user_query'] = user_query

        # Clear previous debug output and response when a new query is submitted
        st.session_state['debug_output'] = ''
        st.session_state['api_response'] = None

        max_attempts = 3
        attempt = 0
        error_message = None  # Initialize error_message
        api_response = None  # Initialize the API response for success tracking

        while attempt < max_attempts:
            attempt += 1
            add_debug_info(f"Attempt {attempt} of {max_attempts}", attempt)

            if not user_query or not st.session_state['selected_api']:
                st.warning("Please select an API and enter a query.")
                break
            else:
                knowledge_base = get_knowledge_base(st.session_state['selected_api'])
                if knowledge_base:
                    # Pass the error_message to extract_parameters
                    params = extract_parameters(user_query, knowledge_base, error_message=error_message)
                    api_result = create_api_call(knowledge_base, params)

                    if api_result["success"]:
                        api_response = api_result["response"]
                        st.session_state['api_response'] = api_response  # Store in session state

                        summary = summarize_response(user_query, api_response)
                        st.write("**Response:**")
                        st.write(summary)
                        break  # Exit the loop on success
                    else:
                        # Update error_message with the latest error
                        error_message = api_result.get("error", "Unknown error")
                        add_debug_info(f"Error encountered on attempt {attempt}: {error_message}", attempt)

                        if attempt == max_attempts:
                            st.error(f"Maximum number of attempts reached. Final error: {error_message}")
                else:
                    st.error("Knowledge base not found or invalid for the selected API.")
                    break  # Exit the loop if knowledge base is invalid

        # Display debug info using expander
        if st.session_state.get('debug_output'):
            with st.expander("Show Debug Output"):
                st.text_area("Debug Output", st.session_state['debug_output'], height=400)

    # Reset the query input field after submission to allow new query
    st.session_state['user_query'] = ''  # Clear user query after submission

# Function to read text from a .docx file
def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Function for 'Manage APIs' tab
def manage_apis_tab():
    # Display available APIs
    api_list = get_api_list()
    if api_list:
        # Create a dataframe-like structure with the API list
        formatted_api_list = [
            {
                "ID": api[0],
                "API Name": api[1],
                "Short Description": api[2],
                "Base URL": api[3]
            } for api in api_list
        ]
        st.dataframe(formatted_api_list)
    else:
        st.info("No APIs available. Please add one using the form above.")

    # Create two columns for the URL input and file uploader
    col1, col2 = st.columns([2, 2])  # Adjust the size of the columns if needed

    with col1:
        uploaded_file = st.file_uploader("Upload API documentation (.docx or .txt)", type=['docx', 'txt'])

    with col2:
        new_api_url = st.text_input("Enter API documentation URL:")

    # Add button below the input fields
    if st.button("Add API"):
        if new_api_url:
            doc_text = fetch_documentation_from_url(new_api_url)
        elif uploaded_file is not None:
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc_text = read_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                doc_text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Please provide either a URL or upload a file.")
            return  # Early return if no input provided

        # Build knowledge base from the provided text
        knowledge_base = build_knowledge_base(doc_text)
        if knowledge_base:
            source = new_api_url if new_api_url else "Uploaded File"
            save_api_to_db(source, knowledge_base)
            st.success(f"Added API: {source}")
            
            # Rerun the app to reflect the changes in the API list
            st.rerun()  # Force a page refresh to update the API list
        else:
            st.error("Failed to build knowledge base. Ensure the API documentation contains the base URL.")

# Function for 'Search APIs' tab
def search_apis_tab():
    search_query = st.text_input("Search for APIs:")
    if search_query:
        api_list = get_api_list()
        search_results = [api for api in api_list if search_query.lower() in api[1].lower()]
        if search_results:
            st.write(f"Search results for '{search_query}':")
            for api in search_results:
                st.write(f"**{api[1]}** - {api[2]}\n_Base URL_: {api[3]}")
        else:
            st.info("No APIs match your search query.")
    else:
        st.info("Enter a search query to find APIs.")

# Function for 'Documentation' tab
def documentation_tab():
    api_list = get_api_list()
    if api_list:
        selected_api = st.selectbox("Select an API for documentation", api_list, format_func=lambda x: f"{x[1]} - {x[2]}")
        if selected_api:
            knowledge_base = get_knowledge_base(selected_api[0])
            if knowledge_base:
                st.write("**Knowledge Base:**")
                st.json(knowledge_base)
            else:
                st.info("Knowledge base not found for the selected API.")
    else:
        st.info("No APIs available. Please add one in the 'Manage APIs' tab.")

if __name__ == "__main__":
    main()
