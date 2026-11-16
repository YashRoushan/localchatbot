import asyncio

import httpx
import pytest

import chatbot


@pytest.fixture
def app_state(monkeypatch):
    clock = [100.0]
    calls = []

    async def model_chat(*, model, messages, options):
        calls.append((model, [item.copy() for item in messages]))
        await asyncio.sleep(0.01)
        return {"message": {"content": f"Answer to {messages[-1]['content']}"}}

    monkeypatch.setattr(chatbot, "sessions", chatbot.SessionStore(ttl=10, clock=lambda: clock[0]))
    monkeypatch.setattr(chatbot.ollama_client, "chat", model_chat)
    monkeypatch.setattr(chatbot, "MODEL", "test-model")
    monkeypatch.setattr(chatbot.limiter, "enabled", False)
    return clock, calls


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=chatbot.app),
        base_url="https://testserver",
    )


async def ask(browser, text):
    response = await browser.post("/chat", json={"text": text})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_curated_intents_skip_the_model(app_state):
    _, calls = app_state
    async with client() as browser:
        assert (await ask(browser, "How do I reset my Dalhousie password?"))["links"] == [
            "https://csid.cs.dal.ca"
        ]
        assert (await ask(browser, "I can't connect to eduroam"))["links"] == [
            "https://wireless.dal.ca"
        ]
        assert (await ask(browser, "Outlook not working"))["links"] == [
            "https://outlook.office.com"
        ]
        assert not calls


@pytest.mark.asyncio
async def test_concurrent_visitors_have_isolated_context_and_cookie(app_state):
    _, calls = app_state
    async with client() as first, client() as second:
        page, _ = await asyncio.gather(first.get("/"), second.get("/"))
        assert first.cookies[chatbot.SESSION_COOKIE] != second.cookies[chatbot.SESSION_COOKIE]
        assert "HttpOnly" in page.headers["Set-Cookie"]
        assert "Secure" in page.headers["Set-Cookie"]
        assert page.headers["Cache-Control"] == "no-store"
        await asyncio.gather(ask(first, "visitor one monitor"), ask(second, "visitor two keyboard"))
        await asyncio.gather(ask(first, "scope A"), ask(second, "scope B"))
        by_question = {
            messages[-1]["content"]: [item["content"] for item in messages[1:]]
            for model, messages in calls if model == "test-model"
        }
        assert by_question["scope A"] == ["visitor one monitor", "Answer to visitor one monitor", "scope A"]
        assert by_question["scope B"] == ["visitor two keyboard", "Answer to visitor two keyboard", "scope B"]


@pytest.mark.asyncio
async def test_same_session_concurrent_turns_remain_ordered(app_state):
    _, calls = app_state
    async with client() as browser:
        await browser.get("/")
        await asyncio.gather(ask(browser, "monitor goes dark"), ask(browser, "external display"))
        earlier = calls[0][1][-1]["content"]
        assert calls[1][1][1]["content"] == earlier
        assert calls[1][1][2]["content"] == f"Answer to {earlier}"


@pytest.mark.asyncio
async def test_reset_and_expiration_discard_history(app_state):
    clock, calls = app_state
    async with client() as browser:
        await ask(browser, "first question")
        assert (await browser.post("/chat/reset")).json() == {"reset": True}
        await ask(browser, "after reset")
        assert len(calls[-1][1]) == 2
        old_cookie = browser.cookies[chatbot.SESSION_COOKIE]
        clock[0] += 11
        await ask(browser, "after expiration")
        assert len(calls[-1][1]) == 2
        assert browser.cookies[chatbot.SESSION_COOKIE] != old_cookie
        assert old_cookie not in chatbot.sessions.entries


@pytest.mark.asyncio
async def test_model_error_redacts_details_and_is_not_saved(app_state, monkeypatch):
    _, calls = app_state
    working_model = chatbot.ollama_client.chat

    async def fail(**_):
        raise chatbot.ResponseError("private upstream details", 500)

    async with client() as browser:
        await browser.get("/")
        monkeypatch.setattr(chatbot.ollama_client, "chat", fail)
        unavailable = await browser.post("/chat", json={"text": "a failing request"})
        assert unavailable.status_code == 503
        assert unavailable.json() == {"detail": "The AI model is temporarily unavailable."}
        monkeypatch.setattr(chatbot.ollama_client, "chat", working_model)
        await ask(browser, "try again")
        assert len(calls[-1][1]) == 2
        assert (await browser.post("/chat", json={"text": "   "})).status_code == 400
