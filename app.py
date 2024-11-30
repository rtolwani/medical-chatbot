from flask import Flask, request, jsonify, render_template
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv
import logging
import sys
import traceback
from werkzeug.utils import secure_filename

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
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'static/podcasts'
    app.secret_key = os.urandom(24)  # for flash messages

    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
                    <title>Ashita Tolwani's Medical Assistant</title>
                    <script src="https://cdn.tailwindcss.com"></script>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
                    <link href="https://unpkg.com/heroicons@2.0.18/outline/microphone.svg" rel="stylesheet">
                    <style>
                        body {
                            font-family: 'Inter', sans-serif;
                        }
                        .tab-active {
                            border-bottom: 2px solid #2563eb;
                            color: #2563eb;
                        }
                        .mic-active {
                            background-color: #EF4444;
                            color: white;
                        }
                        .mic-pulse {
                            animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                        }
                        @keyframes pulse {
                            0%, 100% {
                                opacity: 1;
                            }
                            50% {
                                opacity: .5;
                            }
                        }
                        .podcast-player {
                            width: 100%;
                            max-width: 600px;
                        }
                        .upload-area {
                            border: 2px dashed #CBD5E1;
                            border-radius: 0.5rem;
                            transition: all 0.2s ease;
                        }
                        .upload-area:hover {
                            border-color: #3B82F6;
                            background-color: #F8FAFC;
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
                                             alt="Ashita Tolwani" 
                                             class="w-full h-full object-cover">
                                    </div>
                                    <div class="text-center md:text-left">
                                        <h1 class="text-3xl font-bold text-gray-800">Ashita Tolwani, MD</h1>
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

                                <!-- Welcome Message -->
                                <div class="text-center mb-8">
                                    <p class="text-xl text-gray-700">Hello, I'm Ashita Tolwani MD's AI.<br/>What would you like to discuss today?</p>
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
                                                <button type="button"
                                                        id="micButton"
                                                        class="p-3 border border-gray-300 rounded-lg hover:bg-gray-50 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                                        title="Click to speak">
                                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                                                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                                                    </svg>
                                                </button>
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
                                        
                                        <!-- Upload Area -->
                                        <div class="upload-area p-6 mb-8 text-center">
                                            <input type="file" id="podcastUpload" accept=".mp3,.wav,.m4a,.ogg" class="hidden">
                                            <label for="podcastUpload" class="cursor-pointer">
                                                <div class="text-gray-500 mb-2">
                                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-8 h-8 mx-auto mb-2">
                                                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
                                                    </svg>
                                                    <p class="text-sm">Drag and drop your podcast file here or click to browse</p>
                                                    <p class="text-xs text-gray-400 mt-1">Supported formats: MP3, WAV, M4A, OGG (Max 16MB)</p>
                                                </div>
                                            </label>
                                        </div>

                                        <!-- Podcast List -->
                                        <div id="podcastList" class="space-y-4">
                                            <!-- Podcasts will be loaded here -->
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <script>
                        // Podcast Upload and Player Functionality
                        document.getElementById('podcastUpload').addEventListener('change', async (e) => {
                            const file = e.target.files[0];
                            if (!file) return;

                            const formData = new FormData();
                            formData.append('podcast', file);

                            try {
                                const response = await fetch('/upload_podcast', {
                                    method: 'POST',
                                    body: formData
                                });

                                const data = await response.json();
                                if (data.success) {
                                    loadPodcasts();
                                } else {
                                    alert(data.error || 'Upload failed');
                                }
                            } catch (error) {
                                console.error('Upload error:', error);
                                alert('Failed to upload podcast');
                            }
                        });

                        async function loadPodcasts() {
                            try {
                                const response = await fetch('/get_podcasts');
                                const podcasts = await response.json();
                                
                                const podcastList = document.getElementById('podcastList');
                                podcastList.innerHTML = '';

                                podcasts.forEach(podcast => {
                                    const podcastElement = document.createElement('div');
                                    podcastElement.className = 'bg-gray-50 p-4 rounded-lg';
                                    podcastElement.innerHTML = `
                                        <p class="text-gray-700 font-medium mb-2">${podcast.filename}</p>
                                        <audio controls class="podcast-player">
                                            <source src="${podcast.path}" type="audio/mpeg">
                                            Your browser does not support the audio element.
                                        </audio>
                                    `;
                                    podcastList.appendChild(podcastElement);
                                });
                            } catch (error) {
                                console.error('Failed to load podcasts:', error);
                            }
                        }

                        // Load podcasts when switching to podcast tab
                        document.getElementById('podcastsTab').addEventListener('click', () => {
                            loadPodcasts();
                        });

                        // Drag and drop functionality
                        const uploadArea = document.querySelector('.upload-area');
                        
                        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                            uploadArea.addEventListener(eventName, preventDefaults, false);
                        });

                        function preventDefaults(e) {
                            e.preventDefault();
                            e.stopPropagation();
                        }

                        ['dragenter', 'dragover'].forEach(eventName => {
                            uploadArea.addEventListener(eventName, highlight, false);
                        });

                        ['dragleave', 'drop'].forEach(eventName => {
                            uploadArea.addEventListener(eventName, unhighlight, false);
                        });

                        function highlight(e) {
                            uploadArea.classList.add('border-blue-500', 'bg-blue-50');
                        }

                        function unhighlight(e) {
                            uploadArea.classList.remove('border-blue-500', 'bg-blue-50');
                        }

                        uploadArea.addEventListener('drop', handleDrop, false);

                        function handleDrop(e) {
                            const dt = e.dataTransfer;
                            const file = dt.files[0];
                            
                            const input = document.getElementById('podcastUpload');
                            input.files = dt.files;
                            input.dispatchEvent(new Event('change'));
                        }

                        // Speech Recognition Setup
                        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                        let recognition = null;
                        let isListening = false;

                        if (SpeechRecognition) {
                            recognition = new SpeechRecognition();
                            recognition.continuous = true;
                            recognition.interimResults = true;
                            recognition.lang = 'en-US';

                            recognition.onstart = () => {
                                isListening = true;
                                const micButton = document.getElementById('micButton');
                                micButton.classList.add('mic-active', 'mic-pulse');
                            };

                            recognition.onend = () => {
                                isListening = false;
                                const micButton = document.getElementById('micButton');
                                micButton.classList.remove('mic-active', 'mic-pulse');
                            };

                            recognition.onresult = (event) => {
                                const input = document.getElementById('questionInput');
                                let finalTranscript = '';

                                for (let i = event.resultIndex; i < event.results.length; i++) {
                                    const transcript = event.results[i][0].transcript;
                                    if (event.results[i].isFinal) {
                                        finalTranscript += transcript;
                                    }
                                }

                                if (finalTranscript) {
                                    input.value = finalTranscript;
                                    recognition.stop();
                                }
                            };

                            recognition.onerror = (event) => {
                                console.error('Speech recognition error:', event.error);
                                isListening = false;
                                const micButton = document.getElementById('micButton');
                                micButton.classList.remove('mic-active', 'mic-pulse');
                            };

                            // Add click handler for microphone button
                            document.getElementById('micButton').addEventListener('click', () => {
                                if (!isListening) {
                                    recognition.start();
                                } else {
                                    recognition.stop();
                                }
                            });
                        } else {
                            // Hide mic button if speech recognition is not supported
                            document.getElementById('micButton').style.display = 'none';
                        }

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

    @app.route('/upload_podcast', methods=['POST'])
    def upload_podcast():
        if 'podcast' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['podcast']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({
                'success': True,
                'filename': filename,
                'path': f'/static/podcasts/{filename}'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400

    @app.route('/get_podcasts')
    def get_podcasts():
        podcasts = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if allowed_file(filename):
                podcasts.append({
                    'filename': filename,
                    'path': f'/static/podcasts/{filename}'
                })
        return jsonify(podcasts)

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
