"""LAB_MODEL -> chat model. `codex:<model>` uses the ChatGPT subscription through the Codex backend."""

import json
import os
import time
from pathlib import Path

CODEX_URL = "https://chatgpt.com/backend-api/codex"


def chat_model(spec: str):
    if spec.startswith("codex:"):
        return _codex(spec.removeprefix("codex:"))
    from langchain.chat_models import init_chat_model

    return init_chat_model(spec)  # API-key providers, e.g. openai:gpt-... or anthropic:claude-...


def _codex(model: str):
    # Read-only: Pi owns this login and rotates its refresh token, so we never refresh or copy it.
    path = Path(os.environ.get("CODEX_AUTH_FILE", "~/.pi/agent/auth.json")).expanduser()
    auth = json.loads(path.read_text())["openai-codex"]
    if auth["expires"] / 1000 < time.time():
        raise SystemExit(f"ChatGPT login in {path} has expired; start pi once to refresh it.")
    return _codex_class()(
        model=model,
        base_url=CODEX_URL,
        api_key=auth["access"],
        default_headers={"chatgpt-account-id": auth["accountId"], "OpenAI-Beta": "responses=experimental",
                         "originator": "pi"},
        use_responses_api=True,
        streaming=True,  # the Codex backend only serves streamed responses
        store=False,  # and refuses to store them
    )


def _codex_class():
    from langchain_openai import ChatOpenAI

    class ChatCodex(ChatOpenAI):
        """The Codex backend rejects system-role input and requires `instructions` instead."""

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            system = [i for i in payload["input"] if i.get("role") in ("system", "developer")]
            payload["input"] = [i for i in payload["input"] if i not in system]
            payload["instructions"] = "\n\n".join(_text(i["content"]) for i in system) or "You are a helpful assistant."
            return payload

    return ChatCodex


def _text(content) -> str:
    return content if isinstance(content, str) else "\n".join(c.get("text", "") for c in content)
