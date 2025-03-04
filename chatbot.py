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

# Links embedded directly into the code (replacing links.json)
links_data = [
    {"keyword": "password reset", "link": "https://csid.cs.dal.ca"},
    {"keyword": "email", "link": "https://outlook.office.com"},
    {"keyword": "wifi troubleshooting", "link": "https://wireless.dal.ca"},
    {"keyword": "vpn issues", "link": "https://vpn.dal.ca"},
    {"keyword": "equipment loan", "link": "https://helpdesk.cs.dal.ca"},
    {"keyword": "borrow laptop", "link": "https://helpdesk.cs.dal.ca"},
    {"keyword": "room booking", "link": "https://campusbookings.dal.ca/p/"},
    {"keyword": "printer setup", "link": "https://print.cs.dal.ca/"},
    {"keyword": "building access",
        "link": "https://helpdesk.cs.dal.ca/form/access-request"}
]

# Chat history (Short-term memory)
chat_history = deque(maxlen=5)

FAQ_INTENTS = {
    "password reset": ["reset password", "forgot password", "change password"],
    "Wi-Fi issue": ["wifi problem", "can't connect to eduroam", "internet issue"],
    "email access": ["can't access email", "email login issue", "outlook not working"],
}


def preprocess_input(user_input):
    user_input = user_input.lower().strip()
    user_input = re.sub(r"[^a-zA-Z0-9\s]", "", user_input)

    best_match = None
    highest_score = 0

    for intent, phrases in FAQ_INTENTS.items():
        match, score = process.extractOne(user_input, phrases)
        if score > highest_score:
            highest_score = score
            best_match = intent

    return best_match if highest_score > 75 else user_input


def find_matching_links(user_input):
    matching_links = []
    for entry in links_data:
        if entry["keyword"] in user_input.lower():
            matching_links.append(
                f"<a href='{entry['link']}' target='_blank'>{entry['link']}</a>")
    return matching_links


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
            loadingIndicator.style.display = "block";
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
                loadingIndicator.style.display = "none";
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
@limiter.limit("6/minute")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = preprocess_input(data.get("text", "").strip())

        if not user_input:
            raise HTTPException(
                status_code=400, detail="Input cannot be empty.")

        chat_history.append({"role": "user", "content": user_input})

        # Check for links first and return if matched
        matching_links = find_matching_links(user_input)
        if matching_links:
            link_response = "Here are some helpful links you can check:\n" + \
                "\n".join(matching_links)
            return JSONResponse({"response": link_response})

        system_prompt = """
        You are the Dalhousie University Computer Science Help Desk AI.
        Your role is to assist students, faculty, and staff with IT-related issues.

        Guidelines:
        1. Always provide a link if a relevant one is available.
        2. Keep responses within 2-3 sentences.
        3. Never invent or hallucinate links. If you are unsure, say: "Please check Dalhousie's official IT resources."
        4. If a system-provided link is attached, mention it directly like "You can also check this link: [link]".
        5. Use only official Dalhousie resources.
        """

        conversation = [
            {"role": "system", "content": system_prompt}] + list(chat_history)

        response = ollama.chat(model="llama2:13b", messages=conversation)
        bot_response = response["message"]["content"]

        # Add any matching links to the response
        if matching_links:
            bot_response += "\n\nHere are some helpful links you can check:\n" + \
                "\n".join(matching_links)

        chat_history.append({"role": "assistant", "content": bot_response})

        return JSONResponse({"response": bot_response})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An error occurred: {str(e)}"})
