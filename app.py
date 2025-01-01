from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from openai import OpenAI, OpenAIError
import os
import glob
import json
import logging
import traceback
import sys
import psutil
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
VECTOR_STORE_DIR = "vector_store"
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "embeddings.csv")
EMBEDDINGS_CHUNKS_DIR = os.path.join(VECTOR_STORE_DIR, "embeddings_chunks")
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
    """Calculate cosine similarity between two vectors using numpy for better performance."""
    a_array = np.array(a)
    b_array = np.array(b)
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage - RSS: {mem_info.rss / 1024 / 1024:.2f}MB, VMS: {mem_info.vms / 1024 / 1024:.2f}MB")

def load_embeddings() -> pd.DataFrame:
    """Load and combine all embedding chunks into a single DataFrame."""
    log_memory_usage()
    logger.info("Starting to load embeddings")
    
    # Load and process chunks in batches
    all_chunks = []
    chunk_files = sorted(glob.glob(os.path.join(EMBEDDINGS_CHUNKS_DIR, "embeddings_part_*.csv")))
    
    for chunk_file in chunk_files:
        try:
            logger.info(f"Loading chunk file: {chunk_file}")
            # Read CSV in chunks to reduce memory usage
            for chunk_df in pd.read_csv(chunk_file, chunksize=100):
                # Convert embedding strings to numpy arrays immediately
                chunk_df['embedding'] = chunk_df['embedding'].apply(lambda x: np.array(eval(x)))
                all_chunks.append(chunk_df)
                log_memory_usage()
        except Exception as e:
            logger.error(f"Error loading chunk file {chunk_file}: {str(e)}")
    
    if not all_chunks:
        logger.warning("No embedding chunks found!")
        return pd.DataFrame(columns=['text', 'embedding'])
    
    logger.info("Concatenating all chunks")
    result_df = pd.concat(all_chunks, ignore_index=True)
    log_memory_usage()
    return result_df

def create_app():
    # Initialize Flask app
    app = Flask(__name__, static_url_path='/static', static_folder='static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'static/podcasts'
    app.config['REFERENCES_FOLDER'] = 'references'
    app.secret_key = os.urandom(24)  # for flash messages

    # Store conversation history
    conversation_history = {}

    # Log initial memory usage
    log_memory_usage()
    logger.info("Initializing application")

    # Initialize directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REFERENCES_FOLDER'], exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_CHUNKS_DIR, exist_ok=True)

    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'pdf'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_pdf_references() -> List[Dict]:
        """Process PDF files and update vector store."""
        references = []
        
        # Ensure directories exist
        os.makedirs(app.config['REFERENCES_FOLDER'], exist_ok=True)
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        os.makedirs(EMBEDDINGS_CHUNKS_DIR, exist_ok=True)
        
        references_path = os.path.join(app.config['REFERENCES_FOLDER'])
        pdf_files = glob.glob(os.path.join(references_path, '*.pdf'))
        
        if not pdf_files:
            logger.warning("No PDF files found in references directory")
            return []
        
        logger.info(f"Found PDF files: {pdf_files}")
        chunks_data = []
        batch_size = 10  # Process embeddings in batches
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        logger.info(f"Reading page {page_num + 1} of {len(pdf_reader.pages)}")
                        text += page.extract_text()
                    
                    # Chunk the text
                    chunks = chunk_text(text)
                    logger.info(f"Created {len(chunks)} chunks")
                    
                    # Process embeddings in batches
                    current_batch = []
                    for i, chunk in enumerate(chunks):
                        current_batch.append(chunk)
                        
                        # Process batch when it reaches batch_size or is the last chunk
                        if len(current_batch) == batch_size or i == len(chunks) - 1:
                            logger.info(f"Processing batch of {len(current_batch)} chunks")
                            
                            # Get embeddings for the batch
                            batch_embeddings = []
                            for batch_chunk in current_batch:
                                embedding = get_embedding(batch_chunk)
                                if embedding:
                                    batch_embeddings.append({
                                        'text': batch_chunk,
                                        'embedding': embedding,
                                        'source': os.path.basename(pdf_path)
                                    })
                            
                            chunks_data.extend(batch_embeddings)
                            current_batch = []  # Reset batch
                            
                            # Log progress
                            logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                    
                    references.append({
                        'filename': os.path.basename(pdf_path),
                        'content': text
                    })
                    
                logger.info(f"Successfully processed {pdf_path}")
            except Exception as e:
                error_msg = f"Error processing PDF {pdf_path}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                print(error_msg)
        
        # Save chunks in batches
        if chunks_data:
            logger.info(f"Saving {len(chunks_data)} chunks to vector store")
            try:
                df = pd.DataFrame(chunks_data)
                chunk_size = len(df) // 6 + 1  # Split into 6 parts
                os.makedirs(EMBEDDINGS_CHUNKS_DIR, exist_ok=True)
                
                for i, start_idx in enumerate(range(0, len(df), chunk_size)):
                    end_idx = min(start_idx + chunk_size, len(df))
                    chunk_df = df[start_idx:end_idx]
                    chunk_path = os.path.join(EMBEDDINGS_CHUNKS_DIR, f"embeddings_part_{i+1}.csv")
                    chunk_df.to_csv(chunk_path, index=False)
                    logger.info(f"Saved chunk {i+1} to {chunk_path}")
            except Exception as e:
                error_msg = f"Error saving vector store chunks: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                print(error_msg)
        else:
            logger.warning("No chunks were processed successfully")
        
        return references

    # Initialize vector store on startup
    try:
        logger.info("Initializing vector store...")
        get_pdf_references()
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        logger.error(traceback.format_exc())

    def get_relevant_context(query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant context for a query using vectorized operations."""
        try:
            log_memory_usage()
            logger.info("Starting context search")
            
            # Get query embedding once and convert to numpy array
            query_embedding = np.array(get_embedding(query))
            if query_embedding is None:
                return []
            
            # Load embeddings
            df = load_embeddings()
            if df.empty:
                logger.warning("No embeddings found")
                return []

            # Vectorized similarity calculation
            logger.info("Calculating similarities")
            embeddings_matrix = np.vstack(df['embedding'].values)
            query_embedding = query_embedding.reshape(1, -1)
            
            # Compute similarities in a vectorized way
            similarities = np.dot(embeddings_matrix, query_embedding.T).flatten()
            norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
            similarities = similarities / norms
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Get relevant chunks
            relevant_chunks = []
            for idx in top_indices:
                relevant_chunks.append({
                    'text': df.iloc[idx]['text'],
                    'similarity': float(similarities[idx]),
                    'source': df.iloc[idx]['source'] if 'source' in df.columns else 'unknown'
                })
            
            log_memory_usage()
            return relevant_chunks

        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            logger.error(traceback.format_exc())
            return []

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
            logger.info(f"Received data: {data}")  # Log the received data
            
            # More flexible message extraction
            user_message = None
            if isinstance(data, dict):
                user_message = data.get('message') or data.get('query') or data.get('text')
            
            if not user_message:
                logger.error("No message found in request data")
                return jsonify({"error": "No message provided"}), 400

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
                system_prompt = SYSTEM_PROMPT + "\n\nReference Material:\n" + context_text if relevant_contexts else SYSTEM_PROMPT

                # Use the full conversation history instead of just the current message
                messages = conversation_history[session_id].copy()
                
                # Update the system message with the current context
                messages[0]["content"] = system_prompt

                logger.info(f"Sending request to OpenAI with messages: {messages}")
                
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

            except Exception as e:
                error_msg = f"OpenAI API error: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                return jsonify({"error": error_msg}), 500

        except Exception as e:
            error_msg = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(error_msg)
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
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
