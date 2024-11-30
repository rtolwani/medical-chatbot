from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store the medical context
medical_context = ""

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global medical_context
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            medical_context = extract_text_from_pdf(filepath)
            os.remove(filepath)  # Remove file after processing
            return jsonify({'message': 'File processed successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

@app.route('/query', methods=['POST'])
def query():
    global medical_context
    
    if not medical_context:
        return jsonify({'error': 'No medical reference uploaded yet'}), 400
    
    data = request.get_json()
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    question = data['question']
    
    try:
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Answer questions based on the provided medical reference. If the answer cannot be found in the reference, say 'I cannot answer this based on the provided medical reference.'"},
                {"role": "user", "content": f"""Medical Reference:
{medical_context[:3000]}  # Using first 3000 chars for context window

Question: {question}"""}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        
        answer = response.choices[0].message.content.strip()
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
