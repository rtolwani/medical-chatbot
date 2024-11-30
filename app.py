from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

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
    return render_template('templates/index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        # Create chat completion with OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500
        )

        # Extract the assistant's response
        assistant_response = completion.choices[0].message.content
        
        return jsonify({
            "response": assistant_response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
