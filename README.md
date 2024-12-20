# Medical Chatbot

A Flask-based chatbot that answers medical questions using provided medical reference documents.

## Features
- Upload and process medical reference documents (PDF format)
- Ask medical questions through a chat interface
- Get AI-powered responses based on the uploaded references

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a .env file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## Usage
1. Upload medical reference documents using the upload interface
2. Ask medical questions in the chat interface
3. Get responses based on the uploaded references

## Note
This is a demo application and should not be used as a substitute for professional medical advice.