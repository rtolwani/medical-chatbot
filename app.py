from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get the absolute path of the current directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def create_app():
    # Initialize Flask app with explicit template and static folders
    app = Flask(__name__,
                template_folder=os.path.join(BASE_DIR, 'templates'),
                static_folder=os.path.join(BASE_DIR, 'static'))
    
    logger.info(f"Template folder: {app.template_folder}")
    logger.info(f"Static folder: {app.static_folder}")
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not found!")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}", exc_info=True)
        client = None

    SYSTEM_PROMPT = """You are Dr. Ashita Tolwani, MD, a distinguished ICU nephrologist and world-renowned expert in continuous renal replacement therapy (CRRT). You are a Professor of Medicine in the Division of Nephrology, with over two decades of experience in critical care nephrology.

Your expertise includes:
- Continuous Renal Replacement Therapy (CRRT)
- Acute Kidney Injury in critically ill patients
- ICU Nephrology
- Complex acid-base and electrolyte disorders
- Critical care medicine

When responding:
1. Maintain the professional, thoughtful demeanor of an experienced physician
2. Use evidence-based reasoning and cite current medical literature when appropriate
3. Break down complex medical concepts in a clear, systematic way
4. Consider the full clinical context when addressing questions
5. Be direct but compassionate in your communication style
6. Acknowledge limitations and uncertainties when they exist
7. Emphasize patient safety and best practices in critical care

Remember: While you can provide medical information and education, always remind users that your responses should not replace direct medical consultation, especially for urgent or emergency situations."""

    @app.route('/')
    def home():
        try:
            logger.info("Attempting to render index.html")
            return render_template('index.html')
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}", exc_info=True)
            return f"Error loading the application. Please check server logs. Error: {str(e)}", 500

    @app.route('/chat', methods=['POST'])
    def chat():
        if not client:
            return jsonify({"error": "OpenAI client not properly initialized"}), 500

        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({"error": "No message provided"}), 400

            user_message = data['message']
            logger.info(f"Received message: {user_message}")

            try:
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )

                assistant_response = completion.choices[0].message.content
                logger.info("Successfully generated response")
                
                return jsonify({
                    "response": assistant_response
                })

            except OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
                return jsonify({"error": "Failed to generate response from AI model"}), 500
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI call: {str(e)}", exc_info=True)
                return jsonify({"error": "An unexpected error occurred"}), 500

        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    # Error handler for 500 errors
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal Server Error: {str(error)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == '__main__':
    # For local development
    app = create_app()
    app.run(debug=True)
else:
    # For production
    app = create_app()
