from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import ollama

app = FastAPI()

# HTML template for the web UI
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


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_chatbot():
    return HTMLResponse(content=html_template)


# Chatbot API
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("text", "")

    system_prompt = "You are a Dalhousie University Help Desk AI. Answer questions related to IT support."

    response = ollama.chat(
        model="llama2:13b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    return JSONResponse({"response": response["message"]["content"]})