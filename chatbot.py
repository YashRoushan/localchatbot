import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import ollama
import re
from fuzzywuzzy import process  # For typo handling
from collections import deque  # For maintaining chat history
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()

# Rate limiter: Max 6 requests per minute per user
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please wait and try again later."}
    )

# Load links from links.json


def load_links():
    try:
        with open("links.json", "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading links.json: {e}")
        return {}


links_data = load_links()

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
    <title>Dalhousie Help Desk Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        .chat-container { max-width: 600px; margin: auto; text-align: left; }
        .user-message { background: #0084ff; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; text-align: right; }
        .bot-message { background: #e5e5ea; padding: 10px; border-radius: 5px; margin: 5px 0; text-align: left; }
        .loading { text-align: center; font-size: 14px; color: gray; display: none; }
        input, button { padding: 10px; margin: 10px 0; width: 100%; }
    </style>
</head>
<body>
    <h1>Computer Science Help Desk Chatbot</h1>
    <div class="chat-container" id="chatbox"></div>
    <p class="loading" id="loading">AI is thinking...</p>
    <input type="text" id="userInput" placeholder="Type a message..." onkeypress="handleKeyPress(event)">
    <button onclick="sendMessage()">Send</button>

    <script>
        async function sendMessage() {
            let userInput = document.getElementById("userInput").value;
            if (!userInput.trim()) return;
            
            let chatbox = document.getElementById("chatbox");
            let loadingIndicator = document.getElementById("loading");
            
            chatbox.innerHTML += `<div class="user-message">${userInput}</div>`;
            document.getElementById("userInput").value = "";
            loadingIndicator.style.display = "block"; // Show loading message
            
            try {
                let response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: userInput })
                });
                let result = await response.json();
                chatbox.innerHTML += `<div class="bot-message">${result.response}</div>`;
            } catch (error) {
                chatbox.innerHTML += `<div class="bot-message" style="color:red;">Error fetching response</div>`;
            } finally {
                loadingIndicator.style.display = "none"; // Hide loading message
            }
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


@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return HTMLResponse(content=html_template)


@app.post("/chat")
@limiter.limit("6/minute")  # Limit each user to 6 requests per minute
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = preprocess_input(data.get("text", "").strip())

        if not user_input:
            raise HTTPException(
                status_code=400, detail="Input cannot be empty."
            )

        # Check if the input matches a keyword in links.json
        for link_entry in links_data:
            if link_entry["keyword"] in user_input:
                return JSONResponse({"response": f"Here's a helpful link: {link_entry['link']}"})

        # Append user input to chat history
        chat_history.append({"role": "user", "content": user_input})

        # Generate system prompt with chat history for context
        system_prompt = """
        You are the Dalhousie University Computer Science Help Desk AI.  
Your role is to assist students, faculty, and staff with IT-related issues.

### Guidelines:
1. **Always Provide a Link First:** If a link is available, provide it before giving step-by-step instructions.  
2. **Concise Responses:** Keep answers within **2 to 3 sentences** whenever possible.  
3. **No Hallucinations:** If unsure, say:  
   "I'm here to assist with IT-related issues at Dalhousie University. For non-IT questions, please refer to the appropriate department."  
4. **Encourage Self-Service:** Direct users to forms instead of explaining long steps.  
5. **Official Guidance Only:** Use only official Dalhousie resources.  
        """

        conversation = [
            {"role": "system", "content": system_prompt}] + list(chat_history)

        # Call LLaMA 2 with chat history
        response = ollama.chat(model="llama2:13b", messages=conversation)
        bot_response = response["message"]["content"]

        # Append bot response to history
        chat_history.append({"role": "assistant", "content": bot_response})

        return JSONResponse({"response": bot_response})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing your request."}
        )
