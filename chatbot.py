import asyncio
import os
import re
import secrets
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fuzzywuzzy import process
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

SESSION_COOKIE = "helpdesk_session"
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 1024
MODEL = os.environ.get("OLLAMA_MODEL", "llama2:13b")
ollama_client = AsyncClient(timeout=45)

SYSTEM_PROMPT = (
    "You are an independent IT support demo for Dalhousie University. "
    "Keep responses to two or three sentences. If a relevant resource is known, "
    "use an official Dalhousie URL. Never invent links. When unsure, direct the "
    "visitor to Dalhousie's official IT resources. Never ask for credentials."
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please wait and try again later.",
            "error": "Too many requests. Please wait and try again later.",
        },
    )


LINKS = [
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

FAQ_INTENTS = {
    "password reset": [
        "reset password", "forgot password", "change password", "reset my dalhousie password",
    ],
    "wifi troubleshooting": ["wifi problem", "can't connect to eduroam", "internet issue"],
    "email": ["can't access email", "email login issue", "outlook not working"],
}


@dataclass
class Session:
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=monotonic)
    active: int = 0


class SessionStore:
    def __init__(self, ttl=SESSION_TTL_SECONDS, limit=MAX_SESSIONS, clock=monotonic):
        self.entries = OrderedDict()
        self.ttl = ttl
        self.limit = limit
        self.clock = clock

    @asynccontextmanager
    async def open(self, request: Request, response: Response):
        now = self.clock()
        for expired_token in [
            key for key, entry in self.entries.items()
            if not entry.active and now - entry.last_used >= self.ttl
        ]:
            del self.entries[expired_token]
        token = request.cookies.get(SESSION_COOKIE)
        session = self.entries.get(token)

        if session is None:
            if len(self.entries) >= self.limit:
                for old_token, old_session in self.entries.items():
                    if not old_session.active:
                        del self.entries[old_token]
                        break
                else:
                    raise HTTPException(503, "The chat is busy. Please try again shortly.")
            token = secrets.token_urlsafe(32)
            session = self.entries[token] = Session(last_used=now)

        session.active += 1
        try:
            async with session.lock:
                response.set_cookie(
                    SESSION_COOKIE,
                    token,
                    max_age=self.ttl,
                    httponly=True,
                    secure=request.url.scheme == "https",
                    samesite="lax",
                )
                response.headers["Cache-Control"] = "no-store"
                yield session.history
        finally:
            session.active -= 1
            session.last_used = self.clock()
            self.entries.move_to_end(token)


sessions = SessionStore()


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def find_matching_links(text):
    clean = " ".join(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())
    if not clean:
        return []
    matches = [item["link"] for item in LINKS if item["keyword"] in clean]
    if not matches:
        score, intent = max(
            (process.extractOne(clean, phrases)[1], name)
            for name, phrases in FAQ_INTENTS.items()
        )
        if score > 75:
            matches = [item["link"] for item in LINKS if item["keyword"] == intent]
    return list(dict.fromkeys(matches))


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IT Help Desk Chatbot</title>
  <style>
    * { box-sizing: border-box; }
    body { font: 1rem/1.6 system-ui, sans-serif; color: #152033; max-width: 44rem; margin: 2rem auto; padding: 0 1rem; }
    form { display: flex; flex-wrap: wrap; gap: .6rem; align-items: end; }
    label { display: block; }
    .question { flex: 1 1 15rem; }
    input, button { font: inherit; padding: .55rem; }
    input { width: 100%; }
    button { cursor: pointer; }
    .turn { background: #eef2f7; border-radius: .5rem; padding: .6rem 1rem; margin-block: .5rem; white-space: pre-wrap; overflow-wrap: anywhere; }
    .visitor { background: #e0eeff; }
    #status { min-height: 1.6em; }
  </style>
</head>
<body>
  <main>
    <h1>IT Help Desk Chatbot</h1>
    <p>Independent portfolio demo. Use general IT questions and leave out passwords or other private information.</p>
    <div id="chatbox" role="log" aria-label="Conversation"></div>
    <p id="status" role="status"></p>
    <form id="chat-form">
      <div class="question">
        <label for="question">IT question</label>
        <input id="question" type="text" maxlength="2000" required autocomplete="off">
      </div>
      <button id="send">Send</button>
      <button id="reset" type="button">Reset conversation</button>
    </form>
  </main>
  <script>
    const chatbox = document.querySelector("#chatbox");
    const form = document.querySelector("#chat-form");
    const input = document.querySelector("#question");
    const reset = document.querySelector("#reset");
    const send = document.querySelector("#send");
    const status = document.querySelector("#status");

    function appendTurn(text, visitor = false, links = []) {
      const turn = document.createElement("div");
      turn.className = visitor ? "turn visitor" : "turn";
      turn.textContent = (visitor ? "You — " : "Assistant — ") + text;
      for (const link of links) {
        const line = document.createElement("div");
        const anchor = document.createElement("a");
        anchor.href = link;
        anchor.textContent = link;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        line.append(anchor);
        turn.append(line);
      }
      chatbox.append(turn);
    }

    async function request(path, body) {
      send.disabled = reset.disabled = true;
      status.textContent = path === "chat" ? "Waiting for a response…" : "Resetting…";
      try {
        const response = await fetch(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Request failed. Try again.");
        return result;
      } catch (error) {
        status.textContent = error.message;
        return null;
      } finally {
        send.disabled = reset.disabled = false;
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text || send.disabled) return;
      appendTurn(text, true);
      input.value = "";
      const result = await request("chat", { text });
      if (result) {
        appendTurn(result.links.length ? "Here are some helpful IT resources." : result.response, false, result.links);
        status.textContent = "";
      }
      input.focus();
    });

    reset.addEventListener("click", async () => {
      if (await request("chat/reset", {})) {
        chatbox.replaceChildren();
        status.textContent = "Conversation reset.";
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_chatbot(request: Request):
    response = HTMLResponse(content=HTML)
    async with sessions.open(request, response):
        pass
    return response


@app.post("/chat")
@limiter.limit("6/minute")
async def chat(request: Request, response: Response, data: ChatInput):
    text = data.text.strip()
    if not text:
        raise HTTPException(400, "Input cannot be empty.")

    async with sessions.open(request, response) as history:
        user_message = {"role": "user", "content": text}
        links = find_matching_links(text)
        if links:
            answer = "Here are some helpful links:\n" + "\n".join(links)
        else:
            try:
                result = await ollama_client.chat(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *history, user_message],
                    options={"num_predict": 256},
                )
                answer = result["message"]["content"].strip()
                if not answer:
                    raise ValueError("Empty model response")
            except (ValueError, LookupError, OSError, ResponseError, httpx.HTTPError):
                raise HTTPException(503, "The AI model is temporarily unavailable.") from None

        history.append(user_message)
        history.append({"role": "assistant", "content": answer})
        return {"response": answer, "links": links}


@app.post("/chat/reset")
@limiter.limit("12/minute")
async def reset(request: Request, response: Response):
    async with sessions.open(request, response) as history:
        history.clear()
    return {"reset": True}
