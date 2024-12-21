from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
import logging
import sys
import traceback
from werkzeug.utils import secure_filename
import glob
import PyPDF2

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

SYSTEM_PROMPT = """You are Dr. Ravi Tolwani, DVM PhD, a distinguished veterinarian and AI expert. You are the Associate Vice President at The Rockefeller University, focusing on the intersection of medicine and artificial intelligence.

Your expertise includes:
- Artificial Intelligence in Medicine
- Veterinary Medicine
- Machine Learning Applications in Healthcare

Please provide insights and guidance on questions related to the interface of medicine and artificial intelligence."""

def create_app():
    # Initialize Flask app
    app = Flask(__name__, static_url_path='/static', static_folder='static')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'static/podcasts'
    app.config['REFERENCES_FOLDER'] = 'references'
    app.secret_key = os.urandom(24)  # for flash messages

    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg', 'pdf'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_pdf_references():
        """Get content from all PDF files in the references folder."""
        references = []
        references_path = os.path.join(app.config['REFERENCES_FOLDER'])
        logger.info(f"Looking for PDFs in: {references_path}")
        pdf_files = glob.glob(os.path.join(references_path, '*.pdf'))
        logger.info(f"Found PDF files: {pdf_files}")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Attempting to read PDF: {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        logger.info(f"Reading page {page_num + 1} of {pdf_path}")
                        text += page.extract_text()
                    references.append({
                        'filename': os.path.basename(pdf_path),
                        'content': text
                    })
                    logger.info(f"Successfully read PDF: {pdf_path}")
            except Exception as e:
                logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Total references processed: {len(references)}")
        return references

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
            if not data or 'query' not in data:
                return jsonify({"error": "No message provided"}), 400

            user_message = data['query']
            logger.info(f"Received message: {user_message}")

            # Get PDF references
            references = get_pdf_references()
            reference_context = "\n\nAvailable references:\n"
            for ref in references:
                reference_context += f"From {ref['filename']}:\n{ref['content'][:500]}...\n\n"

            try:
                completion = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT + reference_context},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                assistant_response = completion.choices[0].message.content
                return jsonify({"response": assistant_response})

            except OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return jsonify({"error": "Failed to get response from AI service"}), 500

        except Exception as e:
            logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == '__main__':
    # For local development
    app = create_app()
    port = int(os.getenv('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
