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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
                    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
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
                        .mic-active {
                            background-color: #EF4444;
                            color: white;
                        }
                        .mic-pulse {
                            animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                        }
                        @keyframes pulse {
                            0%, 100% { opacity: 1; }
                            50% { opacity: .5; }
                        }
                        .chat-message {
                            font-size: 1rem;
                            line-height: 1.5;
                            padding: 1rem 1.25rem;
                            border-radius: 0.75rem;
                            margin: 0.75rem 0;
                        }
                        .typing-indicator {
                            display: flex;
                            align-items: center;
                            padding: 1.25rem 1.5rem;
                            border-radius: 1rem;
                            margin: 1rem 0;
                            background: #f3f4f6;
                            width: fit-content;
                        }
                        .typing-indicator span {
                            width: 8px;
                            height: 8px;
                            margin: 0 2px;
                            background-color: #6b7280;
                            border-radius: 50%;
                            display: inline-block;
                            opacity: 0.4;
                        }
                        .typing-indicator span:nth-child(1) {
                            animation: typing 1s infinite;
                        }
                        .typing-indicator span:nth-child(2) {
                            animation: typing 1s infinite 0.2s;
                        }
                        .typing-indicator span:nth-child(3) {
                            animation: typing 1s infinite 0.4s;
                        }
                        @keyframes typing {
                            0%, 100% {
                                transform: translateY(0);
                                opacity: 0.4;
                            }
                            50% {
                                transform: translateY(-4px);
                                opacity: 0.8;
                            }
                        }
                        #questionInput {
                            font-size: 1rem;
                            padding: 0.75rem 1rem;
                        }
                        @media (max-width: 768px) {
                            .chat-message {
                                font-size: 1.25rem;
                                line-height: 1.75;
                                padding: 1.25rem 1.5rem;
                            }
                            #questionInput {
                                font-size: 1.25rem;
                                padding: 1rem 1.25rem;
                            }
                            .header-title {
                                font-size: 2rem !important;
                            }
                            .header-subtitle {
                                font-size: 1.5rem !important;
                            }
                            .header-text {
                                font-size: 1.25rem !important;
                            }
                            .welcome-text {
                                font-size: 1.125rem !important;
                                line-height: 1.4 !important;
                                margin: 2rem 0 !important;
                            }
                            .send-button {
                                padding: 1.25rem 2rem !important;
                                font-size: 1.25rem !important;
                            }
                            .mic-button {
                                padding: 1.25rem !important;
                            }
                            .mic-button svg {
                                width: 2rem !important;
                                height: 2rem !important;
                            }
                            .tab-button {
                                font-size: 1rem !important;
                                padding: 1.25rem 2rem !important;
                            }
                            .expertise-tag {
                                font-size: 1.125rem !important;
                                padding: 0.75rem 1.25rem !important;
                            }
                        }
                        .welcome-text {
                            font-size: 1rem;
                            line-height: 1.4;
                            padding: 0.75rem 1rem;
                            color: #374151;
                        }
                        .tab-button {
                            font-size: 0.875rem;
                            padding: 0.5rem 1rem;
                        }
                        .bio-content {
                            font-size: 1rem;
                            line-height: 1.6;
                            padding: 1.5rem;
                            max-width: 800px;
                            margin: 0 auto;
                            text-align: left;
                        }
                        .bio-content p {
                            margin-bottom: 1.5rem;
                        }
                        @media (max-width: 768px) {
                            .welcome-text {
                                font-size: 1.125rem;
                            }
                            .tab-button {
                                font-size: 1rem;
                            }
                            .bio-content {
                                font-size: 1.125rem;
                                padding: 1rem;
                            }
                        }
                    </style>
                </head>
                <body class="bg-gray-50">
                    <div class="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
                        <!-- Navigation Tabs -->
                        <div class="border-b border-gray-200 bg-white">
                            <div class="container mx-auto px-4">
                                <div class="flex justify-center space-x-8">
                                    <button onclick="showTab('chat')" 
                                            class="tab-button bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors" 
                                            id="chatTab">
                                        Chat
                                    </button>
                                    <button onclick="showTab('podcast')" 
                                            class="tab-button bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors" 
                                            id="podcastTab">
                                        Podcast
                                    </button>
                                    <button onclick="showTab('bio')" 
                                            class="tab-button bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors" 
                                            id="bioTab">
                                        Bio
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div class="container mx-auto px-4 py-6">
                            <div class="max-w-4xl mx-auto">
                                <!-- Header -->
                                <div class="flex flex-col md:flex-row items-center justify-center mb-8 space-y-6 md:space-y-0 md:space-x-8">
                                    <div class="profile-image w-40 h-40 rounded-full overflow-hidden shadow-lg">
                                        <img src="https://static.wixstatic.com/media/06422e_2ad5633eaef843d590f2b44f44a8968f~mv2.png/v1/fill/w_688,h_688,al_c,q_90,usm_0.66_1.00_0.01,enc_avif,quality_auto/Ashita%20Tolwani%20MD.png" 
                                             alt="Ashita Tolwani" 
                                             class="w-full h-full object-cover">
                                    </div>
                                    <div class="text-center md:text-left">
                                        <h1 class="header-title text-4xl font-bold text-gray-800">Ashita Tolwani, MD</h1>
                                        <p class="header-subtitle text-2xl text-gray-600 mt-2">Edwin A. Rutsky Professor of Medicine</p>
                                        <p class="header-text text-xl text-gray-600">Division of Nephrology</p>
                                        <p class="header-text text-xl text-gray-600">University of Alabama at Birmingham</p>
                                        <div class="mt-4 flex flex-wrap justify-center md:justify-start gap-2">
                                            <span class="expertise-tag inline-block bg-blue-100 text-blue-800 text-base px-4 py-2 rounded-full">Nephrology</span>
                                            <span class="expertise-tag inline-block bg-blue-100 text-blue-800 text-base px-4 py-2 rounded-full">Critical Care</span>
                                            <span class="expertise-tag inline-block bg-blue-100 text-blue-800 text-base px-4 py-2 rounded-full">CRRT Expert</span>
                                        </div>
                                    </div>
                                </div>

                                <!-- Welcome Message -->
                                <div class="text-center">
                                    <p class="welcome-text text-gray-700">Hello, I'm Ashita Tolwani MD's AI.<br/>What would you like to discuss today?</p>
                                </div>

                                <!-- Content Sections -->
                                <div id="chatSection" class="transition-opacity duration-200">
                                    <div class="bg-white rounded-lg shadow-lg p-6">
                                        <div id="chatMessages" class="space-y-4 mb-6 max-h-[500px] overflow-y-auto"></div>
                                        <form id="questionForm" class="mt-4">
                                            <div class="flex space-x-4">
                                                <input type="text" 
                                                       id="questionInput" 
                                                       class="flex-1 p-4 text-lg border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500" 
                                                       placeholder="Ask a medical question...">
                                                <button type="button"
                                                        id="micButton"
                                                        class="mic-button p-4 border border-gray-300 rounded-lg hover:bg-gray-50 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                                        title="Click to speak">
                                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-8 h-8">
                                                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                                                    </svg>
                                                </button>
                                                <button type="submit" 
                                                        class="send-button bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-lg font-medium transition-colors duration-200 text-lg">
                                                    Send
                                                </button>
                                            </div>
                                        </form>
                                    </div>
                                </div>

                                <!-- Podcasts Section -->
                                <div id="podcastSection" class="hidden transition-opacity duration-200">
                                    <div class="max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
                                        <h2 class="text-2xl font-bold text-gray-800 mb-4">Podcasts</h2>
                                        
                                        <div class="space-y-6">
                                            <!-- Spotify Podcast -->
                                            <div class="podcast-item max-w-xl mx-auto">
                                                <iframe 
                                                    style="border-radius:12px" 
                                                    src="https://open.spotify.com/embed/episode/6QBjrx18C3guPTFVoW1Hrc?utm_source=generator&theme=0&t=0"
                                                    width="100%" 
                                                    height="175"
                                                    frameBorder="0" 
                                                    allowfullscreen="" 
                                                    allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture">
                                                </iframe>
                                            </div>

                                            <!-- Future Podcasts Placeholder -->
                                            <div class="text-gray-600 italic text-center mt-4 text-sm">
                                                More podcasts coming soon!
                                            </div>
                                        </div>

                                        <!-- Follow on Spotify Button -->
                                        <div class="mt-6 text-center">
                                            <a href="https://open.spotify.com/show/YOUR_SHOW_ID" 
                                               target="_blank" 
                                               class="inline-flex items-center px-4 py-2 bg-[#1DB954] text-white text-sm font-semibold rounded-full hover:bg-[#1ed760] transition-colors">
                                                <svg class="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
                                                    <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.6 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                                                </svg>
                                                Follow on Spotify
                                            </a>
                                        </div>
                                    </div>
                                </div>

                                <!-- Bio Section -->
                                <div id="bioSection" class="hidden">
                                    <div class="bio-content">
                                        <h2 class="text-2xl font-bold mb-4">Ashita Tolwani MD, MSc</h2>
                                        
                                        <p>Dr. Ashita Tolwani is a Professor of Medicine at the University of Alabama at Birmingham (UAB), where she holds the DCI Edwin A. Rutsky Endowed Chair in Nephrology. She earned a Master of Science in Clinical Epidemiology from the Harvard School of Public Health and completed a combined fellowship in nephrology and critical care at UAB, where she has since remained as faculty. Dr. Tolwani serves as Co-Director for Critical Nephrology and founded the UAB Continuous Renal Replacement Therapy (CRRT) Academy, a premier two-day course offering comprehensive training through simulations, along with pre- and post-assessments. The Academy has become an integral part of fellowship training, with many programs requiring annual participation. Attendees hail from across the United States and internationally, with participants from countries including Mexico, the Middle East, and Asia. In addition to her work at UAB, Dr. Tolwani conducts CRRT workshops around the globe.</p>
                                        
                                        <p>Her dedication to teaching and excellence in nephrology has been recognized with numerous institutional and national awards, including the 2016 UAB President's Award for Teaching in the School of Medicine, the 2020 American Society of Nephrology (ASN) Robert G. Narins Award, and the 2023 Vicenza International Critical Care Nephrology Award. She has also been honored by the National Academy of Inventors with the 2023 EnterpreHer Award and was inducted as a member of UAB's Chapter in 2024. Additionally, she received the International AKI and CRRT "Translating Discoveries to Management in AKI" Award in 2024.</p>
                                        
                                        <p>Dr. Tolwani's research focuses on acute kidney injury, ICU nephrology, and CRRT, particularly in citrate anticoagulation. She is the patent holder for a 0.5% citrate replacement solution for CRRT, now used in multiple countries.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <script>
                        // Voice Input Setup with Mobile Support
                        class VoiceInput {
                            constructor() {
                                this.isRecording = false;
                                this.mediaRecorder = null;
                                this.audioChunks = [];
                                this.stream = null;
                                
                                // Setup Web Speech API if available
                                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                                if (SpeechRecognition) {
                                    this.recognition = new SpeechRecognition();
                                    this.recognition.continuous = true;
                                    this.recognition.interimResults = true;
                                    this.recognition.lang = 'en-US';
                                    this.setupSpeechRecognition();
                                } else {
                                    this.recognition = null;
                                }
                            }

                            setupSpeechRecognition() {
                                this.recognition.onstart = () => {
                                    this.updateMicButton(true);
                                };

                                this.recognition.onend = () => {
                                    this.updateMicButton(false);
                                };

                                this.recognition.onresult = (event) => {
                                    let finalTranscript = '';
                                    for (let i = event.resultIndex; i < event.results.length; i++) {
                                        const transcript = event.results[i][0].transcript;
                                        if (event.results[i].isFinal) {
                                            finalTranscript += transcript;
                                        }
                                    }
                                    if (finalTranscript) {
                                        document.getElementById('questionInput').value = finalTranscript;
                                        this.stopRecording();
                                    }
                                };

                                this.recognition.onerror = (event) => {
                                    console.error('Speech recognition error:', event.error);
                                    this.updateMicButton(false);
                                };
                            }

                            updateMicButton(isActive) {
                                const micButton = document.getElementById('micButton');
                                if (isActive) {
                                    micButton.classList.add('mic-active', 'mic-pulse');
                                } else {
                                    micButton.classList.remove('mic-active', 'mic-pulse');
                                }
                            }

                            async startRecording() {
                                try {
                                    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                                    this.isRecording = true;
                                    this.audioChunks = [];
                                    
                                    this.mediaRecorder = new MediaRecorder(this.stream);
                                    this.mediaRecorder.ondataavailable = (event) => {
                                        if (event.data.size > 0) {
                                            this.audioChunks.push(event.data);
                                        }
                                    };

                                    this.mediaRecorder.onstop = async () => {
                                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                                        await this.transcribeAudio(audioBlob);
                                        this.stream.getTracks().forEach(track => track.stop());
                                    };

                                    this.mediaRecorder.start();
                                    this.updateMicButton(true);

                                    // Also start Web Speech API if available
                                    if (this.recognition) {
                                        this.recognition.start();
                                    }
                                } catch (error) {
                                    console.error('Error starting recording:', error);
                                    alert('Unable to access microphone. Please check your permissions.');
                                    this.updateMicButton(false);
                                }
                            }

                            async stopRecording() {
                                if (this.isRecording) {
                                    this.isRecording = false;
                                    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                                        this.mediaRecorder.stop();
                                    }
                                    if (this.recognition) {
                                        this.recognition.stop();
                                    }
                                    this.updateMicButton(false);
                                }
                            }

                            async transcribeAudio(audioBlob) {
                                // Only transcribe if Web Speech API failed or isn't available
                                if (!this.recognition || document.getElementById('questionInput').value === '') {
                                    const formData = new FormData();
                                    formData.append('audio', audioBlob);

                                    try {
                                        const response = await fetch('/transcribe', {
                                            method: 'POST',
                                            body: formData
                                        });
                                        const data = await response.json();
                                        if (data.text) {
                                            document.getElementById('questionInput').value = data.text;
                                        }
                                    } catch (error) {
                                        console.error('Transcription error:', error);
                                    }
                                }
                            }

                            toggleRecording() {
                                if (!this.isRecording) {
                                    this.startRecording();
                                } else {
                                    this.stopRecording();
                                }
                            }
                        }

                        // Initialize voice input
                        const voiceInput = new VoiceInput();

                        // Add click handler for microphone button
                        document.getElementById('micButton').addEventListener('click', () => {
                            voiceInput.toggleRecording();
                        });

                        // Add keyboard shortcut (spacebar) for voice input
                        document.addEventListener('keydown', (e) => {
                            // Only trigger if no input is focused and spacebar is pressed
                            if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
                                e.preventDefault();
                                voiceInput.toggleRecording();
                            }
                        });

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
                        document.getElementById('podcastTab').addEventListener('click', () => {
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

                        // Tab switching functionality
                        function showTab(tabName) {
                            // Hide all sections
                            document.getElementById('chatSection').classList.add('hidden');
                            document.getElementById('podcastSection').classList.add('hidden');
                            document.getElementById('bioSection').classList.add('hidden');
                            
                            // Show selected section
                            document.getElementById(tabName + 'Section').classList.remove('hidden');
                            
                            // Update tab buttons
                            const tabs = document.querySelectorAll('.tab-button');
                            tabs.forEach(tab => {
                                if (tab.textContent.toLowerCase() === tabName) {
                                    tab.classList.remove('bg-gray-500');
                                    tab.classList.add('bg-blue-500');
                                } else {
                                    tab.classList.remove('bg-blue-500');
                                    tab.classList.add('bg-gray-500');
                                }
                            });
                        }

                        function addMessage(type, content) {
                            const messages = document.getElementById('chatMessages');
                            const div = document.createElement('div');
                            div.className = `chat-message ${
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

                        function addTypingIndicator() {
                            const messages = document.getElementById('chatMessages');
                            const indicator = document.createElement('div');
                            indicator.className = 'typing-indicator mr-12';
                            indicator.id = 'typingIndicator';
                            indicator.innerHTML = `
                                <span></span>
                                <span></span>
                                <span></span>
                            `;
                            messages.appendChild(indicator);
                            messages.scrollTop = messages.scrollHeight;
                        }

                        function removeTypingIndicator() {
                            const indicator = document.getElementById('typingIndicator');
                            if (indicator) {
                                indicator.remove();
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

                            // Add typing indicator
                            addTypingIndicator();
                            
                            try {
                                const response = await fetch('/chat', {
                                    method: 'POST',
                                    headers: {
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({ message }),
                                });
                                
                                // Remove typing indicator
                                removeTypingIndicator();

                                const data = await response.json();
                                if (data.error) {
                                    addMessage('error', data.error);
                                } else {
                                    addMessage('assistant', data.response);
                                }
                            } catch (error) {
                                // Remove typing indicator
                                removeTypingIndicator();
                                addMessage('error', 'Failed to get response');
                            }
                        });
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
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT + "\n\nFormat your responses with clear sections and proper spacing. Use clean formatting with line breaks between paragraphs. Use plain text without any special characters or markdown formatting. For sections, use plain text headers without ###. For lists, use simple hyphens (-) or numbers (1.). Add empty lines between all paragraphs and sections for readability."},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )

                assistant_response = completion.choices[0].message.content
                
                # Format the response
                # Remove any asterisks and markdown formatting
                formatted_response = assistant_response.replace('*', '').replace('###', '')
                
                # Split into paragraphs and ensure proper spacing
                paragraphs = [p.strip() for p in formatted_response.split('\n') if p.strip()]
                formatted_response = '\n\n'.join(paragraphs)
                
                # Add extra spacing after periods that end sentences
                formatted_response = formatted_response.replace('. ', '.\n\n')
                
                # Add spacing after bullet points while preserving list structure
                lines = formatted_response.split('\n')
                formatted_lines = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        if line.startswith('-') or line.startswith('1.') or line.startswith('2.'):
                            # Don't add extra space before list items
                            formatted_lines.append(line)
                        else:
                            # Add extra space before non-list items
                            if i > 0 and not lines[i-1].strip().startswith('-') and not any(lines[i-1].strip().startswith(str(n) + '.') for n in range(1, 10)):
                                formatted_lines.append('')
                            formatted_lines.append(line)
                
                formatted_response = '\n'.join(formatted_lines)
                
                # Clean up any excessive newlines
                while '\n\n\n' in formatted_response:
                    formatted_response = formatted_response.replace('\n\n\n', '\n\n')
                
                # Ensure sections are properly spaced
                sections = ['Key Features', 'Indications', 'Types', 'How']
                for section in sections:
                    formatted_response = formatted_response.replace(f'\n{section}', f'\n\n{section}')
                
                return jsonify({"response": formatted_response})

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

    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        try:
            file = request.files['audio']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Transcribe audio file using OpenAI API
            try:
                completion = client.chat.completions.create(
                    model="whisper-1",
                    messages=[
                        {"role": "user", "content": f"Transcribe {filename}"},
                    ],
                    temperature=0.7,
                    max_tokens=500
                )

                transcription = completion.choices[0].message.content
                
                logger.info("Successfully transcribed audio")
                
                return jsonify({
                    "text": transcription
                })

            except OpenAIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({"error": "Failed to transcribe audio"}), 500
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
