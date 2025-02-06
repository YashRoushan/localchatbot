from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import ollama
import re
from fuzzywuzzy import process  # For typo handling
from collections import deque  # For maintaining chat history

app = FastAPI()

# Store chat history (Short-term memory)
chat_history = deque(maxlen=5)  # Keeps the last 5 messages

# IT Support Intent Detection
FAQ_INTENTS = {
    "password reset": ["reset password", "forgot password", "change password"],
    "Wi-Fi issue": ["wifi problem", "can't connect to eduroam", "internet issue"],
    "email access": ["can't access email", "email login issue", "outlook not working"],
}

# Function to clean and preprocess user input


def preprocess_input(user_input):
    user_input = user_input.lower().strip()
    # Remove special characters
    user_input = re.sub(r"[^a-zA-Z0-9\s]", "", user_input)

    # Fuzzy match to detect common IT support queries
    best_match = None
    highest_score = 0
    for intent, phrases in FAQ_INTENTS.items():
        match, score = process.extractOne(user_input, phrases)
        if score > highest_score:
            highest_score = score
            best_match = intent

    return best_match if highest_score > 75 else user_input


# HTML template for chatbot UI
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local AI Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        .chat-container { max-width: 600px; margin: auto; text-align: left; }
        .user-message { background: #0084ff; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; text-align: right; }
        .bot-message { background: #e5e5ea; padding: 10px; border-radius: 5px; margin: 5px 0; text-align: left; }
        input, button { padding: 10px; margin: 10px 0; width: 100%; }
    </style>
</head>
<body>
    <h1>Dalhousie Help Desk Chatbot</h1>
    <div class="chat-container" id="chatbox"></div>
    <input type="text" id="userInput" placeholder="Type a message..." onkeypress="handleKeyPress(event)">
    <button onclick="sendMessage()">Send</button>

    <script>
        async function sendMessage() {
            let userInput = document.getElementById("userInput").value;
            if (!userInput.trim()) return;
            
            let chatbox = document.getElementById("chatbox");
            chatbox.innerHTML += `<div class="user-message">${userInput}</div>`;
            document.getElementById("userInput").value = "";

            let response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: userInput })
            });

            let result = await response.json();
            chatbox.innerHTML += `<div class="bot-message">${result.response}</div>`;
        }

        function handleKeyPress(event) {
            if (event.key === "Enter") {
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

# Serve the HTML page at `/`


@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return HTMLResponse(content=html_template)

# Chatbot API for handling user messages


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = preprocess_input(data.get("text", ""))  # Apply preprocessing

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Generate system prompt with chat history for context
    system_prompt = """
        You are a Dalhousie University Help Desk AI.
        - Answer ONLY questions related to Dalhousie IT support (Wi-Fi, password resets, email issues, software access).
        - If a question is not IT-related, politely say you cannot answer.
        - Be concise and provide direct solutions.
        - If needed, suggest the user contact Dalhousie IT at helpdesk@cs.dal.ca.
    """
    conversation = [
        {"role": "system", "content": system_prompt}] + list(chat_history)

    # Call LLaMA 2 with chat history
    response = ollama.chat(model="llama2:13b", messages=conversation)

    # Save bot response in history
    chat_history.append(
        {"role": "assistant", "content": response["message"]["content"]})

    return JSONResponse({"response": response["message"]["content"]})
