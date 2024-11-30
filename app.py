from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    
    # Log important directories
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"App root path: {app.root_path}")
    logger.info(f"Template folder path: {app.template_folder}")
    
    if os.path.exists(app.template_folder):
        logger.info(f"Templates directory contents: {os.listdir(app.template_folder)}")
    else:
        logger.error(f"Templates directory does not exist: {app.template_folder}")
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key not found!")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        logger.error(traceback.format_exc())
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
            template_path = os.path.join(app.template_folder, 'index.html')
            logger.info(f"Full template path: {template_path}")
            logger.info(f"Template exists: {os.path.exists(template_path)}")
            
            if not os.path.exists(template_path):
                logger.error(f"Template file not found at: {template_path}")
                # Create a basic template if missing
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dr. Tolwani's Medical Assistant</title>
                    <script src="https://cdn.tailwindcss.com"></script>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
                    <style>
                        body {
                            font-family: 'Inter', sans-serif;
                        }
                        .tab-active {
                            border-bottom: 2px solid #2563eb;
                            color: #2563eb;
                        }
                    </style>
                </head>
                <body class="bg-gray-50">
                    <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
                        <!-- Navigation Tabs -->
                        <div class="border-b border-gray-200 bg-white">
                            <div class="container mx-auto px-4">
                                <div class="flex justify-center space-x-8">
                                    <button onclick="switchTab('chat')" 
                                            class="tab-active px-4 py-4 text-sm font-medium transition-colors duration-200" 
                                            id="chatTab">
                                        Chat
                                    </button>
                                    <button onclick="switchTab('podcasts')" 
                                            class="px-4 py-4 text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors duration-200" 
                                            id="podcastsTab">
                                        Podcasts
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div class="container mx-auto px-4 py-8">
                            <div class="max-w-4xl mx-auto">
                                <!-- Header -->
                                <div class="flex flex-col md:flex-row items-center justify-center mb-8 space-y-4 md:space-y-0 md:space-x-8">
                                    <div class="w-48 h-48 rounded-full overflow-hidden shadow-lg">
                                        <img src="https://static.wixstatic.com/media/06422e_2ad5633eaef843d590f2b44f44a8968f~mv2.png/v1/fill/w_688,h_688,al_c,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/Ashita%20Tolwani%20MD.png" 
                                             alt="Dr. Ashita Tolwani" 
                                             class="w-full h-full object-cover">
                                    </div>
                                    <div class="text-center md:text-left">
                                        <h1 class="text-3xl font-bold text-gray-800">Dr. Ashita Tolwani, MD</h1>
                                        <p class="text-lg text-gray-600 mt-2">Professor of Medicine</p>
                                        <p class="text-gray-600">Division of Nephrology</p>
                                        <p class="text-gray-600">University of Alabama at Birmingham</p>
                                        <div class="mt-4">
                                            <span class="inline-block bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full mr-2 mb-2">Nephrology</span>
                                            <span class="inline-block bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full mr-2 mb-2">Critical Care</span>
                                            <span class="inline-block bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full mb-2">CRRT Expert</span>
                                        </div>
                                    </div>
                                </div>

                                <!-- Content Sections -->
                                <div id="chatSection" class="transition-opacity duration-200">
                                    <div class="bg-white rounded-lg shadow-lg p-6">
                                        <div id="chatMessages" class="space-y-4 mb-6 max-h-[500px] overflow-y-auto"></div>
                                        <form id="questionForm" class="mt-4">
                                            <div class="flex space-x-4">
                                                <input type="text" 
                                                       id="questionInput" 
                                                       class="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" 
                                                       placeholder="Ask a medical question...">
                                                <button type="submit" 
                                                        class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                                                    Send
                                                </button>
                                            </div>
                                        </form>
                                    </div>
                                </div>

                                <!-- Podcasts Section -->
                                <div id="podcastsSection" class="hidden transition-opacity duration-200">
                                    <div class="bg-white rounded-lg shadow-lg p-6">
                                        <h2 class="text-2xl font-bold text-gray-800 mb-4">Podcasts</h2>
                                        <p class="text-gray-600">Coming soon! Dr. Tolwani's medical podcasts will be available here.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <script>
                        // Tab switching functionality
                        function switchTab(tab) {
                            const chatSection = document.getElementById('chatSection');
                            const podcastsSection = document.getElementById('podcastsSection');
                            const chatTab = document.getElementById('chatTab');
                            const podcastsTab = document.getElementById('podcastsTab');

                            if (tab === 'chat') {
                                chatSection.classList.remove('hidden');
                                podcastsSection.classList.add('hidden');
                                chatTab.classList.add('tab-active');
                                chatTab.classList.remove('text-gray-500');
                                podcastsTab.classList.remove('tab-active');
                                podcastsTab.classList.add('text-gray-500');
                            } else {
                                chatSection.classList.add('hidden');
                                podcastsSection.classList.remove('hidden');
                                podcastsTab.classList.add('tab-active');
                                podcastsTab.classList.remove('text-gray-500');
                                chatTab.classList.remove('tab-active');
                                chatTab.classList.add('text-gray-500');
                            }
                        }

                        document.getElementById('questionForm').addEventListener('submit', async (e) => {
                            e.preventDefault();
                            const input = document.getElementById('questionInput');
                            const message = input.value.trim();
                            if (!message) return;
                            
                            // Add user message
                            addMessage('user', message);
                            input.value = '';
                            
                            try {
                                const response = await fetch('/chat', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({ message }),
                                });
                                
                                const data = await response.json();
                                if (data.error) {
                                    addMessage('error', data.error);
                                } else {
                                    addMessage('assistant', data.response);
                                }
                            } catch (error) {
                                addMessage('error', 'Failed to get response');
                            }
                        });
                        
                        function addMessage(type, content) {
                            const messages = document.getElementById('chatMessages');
                            const div = document.createElement('div');
                            div.className = `p-4 rounded-lg ${
                                type === 'user' 
                                    ? 'bg-blue-100 text-blue-900 ml-12' 
                                    : type === 'error' 
                                        ? 'bg-red-100 text-red-900' 
                                        : 'bg-gray-100 text-gray-900 mr-12'
                            }`;
                            div.textContent = content;
                            messages.appendChild(div);
                            messages.scrollTop = messages.scrollHeight;
                        }
                    </script>
                </body>
                </html>
                """
            
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
                logger.error(f"OpenAI API error: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": "Failed to generate response from AI model"}), 500
            except Exception as e:
                logger.error(f"Unexpected error during OpenAI call: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": "An unexpected error occurred"}), 500

        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": "Internal server error"}), 500

    return app

if __name__ == '__main__':
    # For local development
    app = create_app()
    app.run(debug=True)
else:
    # For production
    app = create_app()
