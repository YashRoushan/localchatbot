# Local IT Help Desk Chatbot

FastAPI serves a browser chat and a JSON API. It matches common campus IT intents to curated resources and uses Ollama for open-ended questions.

## Local setup

Install Python 3.12 and Ollama. Start Ollama, pull the model you intend to use, then run:

```sh
python -m pip install -r requirements.txt
ollama pull llama2:13b
uvicorn chatbot:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Set `OLLAMA_MODEL` to select another installed model and `OLLAMA_HOST` for a different Ollama endpoint. The default model remains `llama2:13b`.

The API accepts `POST /chat` with a JSON `text` field and returns `response` plus a `links` array. Retain the session cookie on subsequent requests. `POST /chat/reset` clears the current session. Short-term context stays in a single process and expires after 30 idle minutes. The service retains up to five turns per session and 1,024 concurrent retained sessions. Its browser UI renders visitor and model output as text.

## Hosting

The Dockerfile packages the FastAPI UI and API. Run one worker behind an HTTPS reverse proxy and set `OLLAMA_HOST` to a privately reachable Ollama instance with the selected model already installed. Keep Ollama off the public network. The public endpoint requires HTTPS to set a Secure session cookie. Conversation memory is process-local, so a restart clears it. Use a shared session and rate-limit store before running multiple replicas.

## Tests

```sh
python -m pip install pytest pytest-asyncio
python -m pytest -q
```

Tests use an in-process API transport and a mocked model.
