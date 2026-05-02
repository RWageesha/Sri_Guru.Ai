# Sri Guru AI

Streamlit app that chats with local models via Ollama.

## Requirements
- Python 3.9+
- Ollama installed and running

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Pull models:
   ```
   ollama pull llama3
   ollama pull gemma
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Notes
- Ollama must be running at http://localhost:11434.
