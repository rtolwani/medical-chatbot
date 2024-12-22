from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from openai import OpenAI, OpenAIError
import os
import glob
import json
import logging
import traceback
import sys
from dotenv import load_dotenv
import PyPDF2
import pandas as pd
import numpy as np
from typing import List, Dict
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "vector_store.csv"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """You are Dr. Ravi Tolwani, DVM PhD, a distinguished veterinarian and AI expert. You are the Associate Vice President at The Rockefeller University, focusing on the intersection of medicine and artificial intelligence.

Your expertise includes:
- Artificial Intelligence in Medicine
- Veterinary Medicine
- Machine Learning Applications in Healthcare

Background:
Ravi holds a DVM from Auburn University, a Ph.D. in molecular pathology from the University of Alabama School of Medicine, and an MSx from Stanford Graduate School of Business.

Format your responses using this structure and provide comprehensive details:

### Overview
Provide a thorough explanation of the condition including:
- Precise definition with pathophysiology
- Biochemical mechanisms involved
- Classification if applicable
- Epidemiology and risk factors

### Key Points
- **Etiology**: List all major causes with brief explanations
- **Clinical Manifestations**: Detailed symptoms and signs
- **Diagnostic Criteria**: Specific lab values, imaging findings
- **Pathophysiology**: Key mechanisms and progression
- Use **bold** for important terms and values

### Treatment
- **First-line treatments**
- **Alternative options**
- **Supportive care**
- Provide medication dosages and frequencies
- Specify time frames for interventions"""

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()
    
    while start < len(text):
        end = start + CHUNK_SIZE
        
        if end < len(text):
            # Try to find a good breaking point
            last_period = text[end-CHUNK_OVERLAP:end].rfind('. ')
            if last_period != -1:
                end = end - CHUNK_OVERLAP + last_period + 2
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI's API."""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0

def create_app():
    # Initialize Flask app
    app = Flask(__name__, static_url_path='/static', static_folder='static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'static/podcasts'
    app.config['REFERENCES_FOLDER'] = 'references'
    app.secret_key = os.urandom(24)  # for flash messages

    # Store conversation history
    conversation_history = {}

    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'pdf'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_pdf_references() -> List[Dict]:
        """Process PDF files and update vector store."""
        references = []
        references_path = os.path.join(app.config['REFERENCES_FOLDER'])
        pdf_files = glob.glob(os.path.join(references_path, '*.pdf'))
        
        # Check if vector store exists and is up to date
        if os.path.exists(VECTOR_STORE_PATH):
            store_modified = os.path.getmtime(VECTOR_STORE_PATH)
            files_modified = max([os.path.getmtime(f) for f in pdf_files]) if pdf_files else 0
            
            if store_modified > files_modified:
                logger.info("Using cached vector store")
                return []
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        chunks_data = []
        
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        text += page.extract_text()
                    
                    # Chunk the text
                    chunks = chunk_text(text)
                    
                    # Get embeddings for chunks
                    for chunk in chunks:
                        embedding = get_embedding(chunk)
                        if embedding:
                            chunks_data.append({
                                'text': chunk,
                                'embedding': embedding,
                                'source': os.path.basename(pdf_path)
                            })
                    
                    references.append({
                        'filename': os.path.basename(pdf_path),
                        'content': text
                    })
                    
                logger.info(f"Processed {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        
        # Save to CSV
        if chunks_data:
            df = pd.DataFrame(chunks_data)
            df.to_csv(VECTOR_STORE_PATH, index=False)
            logger.info(f"Saved {len(chunks_data)} chunks to vector store")
        
        return references

    def get_relevant_context(query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant context for a query."""
        if not os.path.exists(VECTOR_STORE_PATH):
            logger.warning("Vector store not found")
            return []
        
        # Load vector store
        df = pd.read_csv(VECTOR_STORE_PATH)
        if df.empty:
            return []
        
        # Convert string representations of lists back to actual lists
        df['embedding'] = df['embedding'].apply(eval)
        
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = df['embedding'].apply(lambda x: cosine_similarity(query_embedding, x))
        df['similarity'] = similarities
        
        # Get top k results
        results = df.nlargest(top_k, 'similarity')
        
        return results[['text', 'source', 'similarity']].to_dict('records')

    @app.route('/')
    def home():
        try:
            return render_template('index.html')
        except Exception as e:
            error_msg = f"Error rendering template: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg, 500

    @app.route('/chat', methods=['POST'])
    def chat():
        if not client:
            return jsonify({"error": "OpenAI client not properly initialized"}), 500

        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({"error": "No message provided"}), 400

            user_message = data['message']
            session_id = data.get('session_id', 'default')
            logger.info(f"Received message: {user_message} for session: {session_id}")

            # Initialize conversation history for new sessions
            if session_id not in conversation_history:
                conversation_history[session_id] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]

            # Add user message to history
            conversation_history[session_id].append({"role": "user", "content": user_message})

            try:
                # Get relevant context from vector store
                relevant_contexts = get_relevant_context(user_message)
                context_text = "\n\n".join([
                    f"[From {ctx['source']}]\n{ctx['text']}"
                    for ctx in relevant_contexts
                ])

                # Construct the prompt with context
                system_prompt = SYSTEM_PROMPT + "\n\nReference Material:\n" + context_text

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]

                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )

                assistant_response = completion.choices[0].message.content
                
                # Add assistant response to history
                conversation_history[session_id].append({"role": "assistant", "content": assistant_response})
                
                # Keep only last 10 messages to manage context length
                if len(conversation_history[session_id]) > 11:  # system prompt + 10 messages
                    conversation_history[session_id] = [
                        conversation_history[session_id][0]  # Keep system prompt
                    ] + conversation_history[session_id][-10:]  # Keep last 10 messages
                
                return jsonify({
                    "response": assistant_response,
                    "context": relevant_contexts
                })

            except OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return jsonify({"error": "Failed to get response from AI service"}), 500

        except Exception as e:
            logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                return jsonify({'message': 'File uploaded successfully', 'filename': filename})
            return jsonify({'error': 'File type not allowed'}), 400
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    @app.route('/audio/<filename>')
    def serve_audio(filename):
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        except Exception as e:
            logger.error(f"Error serving audio: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    # For local development
    app = create_app()
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
