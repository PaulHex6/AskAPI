
# ⚡ AskAPI

AskAPI is a Streamlit app that lets you interact with any API using natural language. The app allows you to add APIs, query them, and manage them from a user-friendly interface.

## Features

- Fetch and parse API documentation from a URL
- Convert API documentation into a structured knowledge base using OpenAI API
- Query APIs with natural language and extract relevant parameters
- Create and send API requests based on the knowledge base
- View and manage multiple APIs

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AskAPI.git
   cd AskAPI
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the root directory of your project.
   - Add your OpenAI API key to the file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Use the app to interact with APIs by entering a query, selecting an API, and viewing the responses.

## License
MIT License
