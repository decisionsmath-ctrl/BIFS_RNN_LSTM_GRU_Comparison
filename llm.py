"""
Small helper to call Groq or Google Gemini over plain HTTPS (no extra SDKs).
The API key is sent only in a request header, never in the URL, and is never logged.
"""
import requests

TIMEOUT = 60
GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

_GROQ_SKIP = ("whisper", "tts", "guard", "embed", "orpheus", "transcribe")
_GEMINI_SKIP = ("embedding", "imagen", "tts", "image", "veo", "aqa", "live", "audio")


class LLMError(Exception):
    """Readable error that is safe to show to the user."""


def _json(resp: requests.Response) -> dict:
    if resp.status_code in (401, 403):
        raise LLMError("The API key was rejected (401/403). Check that the key is correct, "
                       "active, and allowed to use this model.")
    if resp.status_code == 429:
        raise LLMError("Rate limit or quota reached (429). Wait a minute, or choose a smaller model.")
    if resp.status_code == 404:
        raise LLMError("Model not found (404). Pick another model from the list.")
    if not resp.ok:
        raise LLMError(f"Provider error HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError:
        raise LLMError("The provider returned a non-JSON response.")


def list_models(provider: str, api_key: str) -> list:
    """Return the chat-capable model ids currently available for this key."""
    if not api_key:
        raise LLMError("No API key provided.")
    try:
        if provider == "Groq":
            data = _json(requests.get(f"{GROQ_BASE}/models",
                                      headers={"Authorization": f"Bearer {api_key}"},
                                      timeout=TIMEOUT))
            ids = [m["id"] for m in data.get("data", []) if m.get("active", True)]
            return sorted(i for i in ids if not any(s in i.lower() for s in _GROQ_SKIP))
        if provider == "Gemini":
            out, token = [], None
            for _ in range(5):                                   # pagination
                params = {"pageSize": 200}
                if token:
                    params["pageToken"] = token
                data = _json(requests.get(f"{GEMINI_BASE}/models",
                                          headers={"x-goog-api-key": api_key},
                                          params=params, timeout=TIMEOUT))
                for m in data.get("models", []):
                    name = m.get("name", "").split("/", 1)[-1]
                    if ("generateContent" in m.get("supportedGenerationMethods", [])
                            and name.startswith("gemini")
                            and not any(s in name.lower() for s in _GEMINI_SKIP)):
                        out.append(name)
                token = data.get("nextPageToken")
                if not token:
                    break
            return sorted(set(out))
    except requests.RequestException as e:
        raise LLMError(f"Network problem: {type(e).__name__}")
    raise LLMError(f"Unknown provider: {provider}")


def chat(provider: str, api_key: str, model: str, system: str, messages: list,
         temperature: float = 0.3, max_tokens: int = 2048) -> str:
    """messages = [{'role': 'user'|'assistant', 'content': str}, ...]. Returns the reply text."""
    if not api_key:
        raise LLMError("No API key provided.")
    if not model:
        raise LLMError("No model selected.")
    try:
        if provider == "Groq":
            payload = {"model": model,
                       "messages": [{"role": "system", "content": system}] + list(messages),
                       "temperature": temperature,
                       "max_completion_tokens": max_tokens}
            data = _json(requests.post(f"{GROQ_BASE}/chat/completions", json=payload,
                                       headers={"Authorization": f"Bearer {api_key}"},
                                       timeout=TIMEOUT))
            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise LLMError("Unexpected response format from Groq.")
            return (text or "").strip() or "(empty reply: try again or choose another model)"

        if provider == "Gemini":
            contents = [{"role": "user" if m["role"] == "user" else "model",
                         "parts": [{"text": m["content"]}]} for m in messages]
            payload = {"systemInstruction": {"parts": [{"text": system}]},
                       "contents": contents,
                       "generationConfig": {"temperature": temperature,
                                            "maxOutputTokens": max_tokens}}
            data = _json(requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                                       json=payload,
                                       headers={"x-goog-api-key": api_key},
                                       timeout=TIMEOUT))
            cands = data.get("candidates") or []
            if not cands:
                reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
                raise LLMError(f"Gemini returned no answer ({reason}).")
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts).strip()
            return text or "(empty reply: try again or choose another model)"
    except requests.RequestException as e:
        raise LLMError(f"Network problem: {type(e).__name__}")
    raise LLMError(f"Unknown provider: {provider}")
