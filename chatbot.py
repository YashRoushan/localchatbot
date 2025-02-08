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
        .user-message { background: #0084ff; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; text-align: right; } //#FFD400
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

        //Refresh page if inactive for 1 minute
        let inactiveTime = 0;
        function resetInactiveTime() {
            inactiveTime = 0;
        }

        function incrementInactiveTime() {
            inactiveTime += 1;
            if (inactiveTime >= 600) { // 1 minute
                location.reload();
            }
        }

        document.addEventListener("mousemove", resetInactiveTime);
        document.addEventListener("keypress", resetInactiveTime);
        setInterval(incrementInactiveTime, 1000); // Check every second

    </script>
</body>
</html>
"""

# Serve the HTML page at `/`


@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return HTMLResponse(content=html_template)

# Chatbot API with rate limiting and improved response formatting


@app.post("/chat")
@limiter.limit("6/minute")  # Limit each user to 6 requests per minute
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = preprocess_input(data.get("text", "").strip())

        if not user_input:
            raise HTTPException(
                status_code=400, detail="Input cannot be empty.")

        # Append user input to chat history
        chat_history.append({"role": "user", "content": user_input})

        # Generate system prompt with chat history for context
        system_prompt = """
        You are a highly knowledgeable and professional Goldberg Computer Science (Dalhousie University) Help Desk AI.
        Your job is to assist users with IT-related issues related to the Computer Science department.

        ### Guidelines:
        1. **Scope:** Answer ONLY IT-related questions (e.g., Wi-Fi, password resets, software access).
        2. **Clarity:** Responses must be clear, structured, and avoid unnecessary complexity.
        3. **Conciseness:** Keep responses short and direct unless a detailed explanation is required.
        4. **No Hallucinations:** If a question is outside your expertise, say:
        "I'm here to assist with IT-related issues at Dalhousie University. For non-IT questions, please refer to the appropriate department."
        5. **Guidance & Links:** When possible, provide official Dalhousie links for self-help.
        6. **Follow-up Questions:** If needed, prompt the user for clarification instead of assuming.
        """

        conversation = [
            {"role": "system", "content": system_prompt}] + list(chat_history)

        # Call LLaMA 2 with chat history
        response = ollama.chat(model="llama2:13b", messages=conversation)

        bot_response = response["message"]["content"]

        # Format specific responses for clarity
        if "password reset" in user_input:
            bot_response = "**Password Reset Instructions:**\n- Visit: [Dalhousie Password Reset](https://csid.cs.dal.ca)\n- If locked out, contact: helpdesk@cs.dal.ca"

        elif "wifi" in user_input or "internet" in user_input:
            bot_response = "**Wi-Fi Troubleshooting:**\n1. Ensure you're connecting to **Eduroam/Dalhousie** using your NetID@dal.ca.\n2. If you forgot your credentials, reset them at https://password.dal.ca.\n3. Still not working? Contact helpdesk@cs.dal.ca."

        # Append bot response to history
        chat_history.append({"role": "assistant", "content": bot_response})

        return JSONResponse({"response": bot_response})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred while processing your request."}
        )
