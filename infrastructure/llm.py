import os
from pathlib import Path

from litellm import completion
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PROVIDER = os.getenv("PROVIDER", "").strip().lower()
MODEL = os.getenv("MODEL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()


YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
SUPPORTED_PROVIDERS = {"openai", "anthropic", "yandex"}


def _validate_settings():
    settings = {
        "PROVIDER": PROVIDER,
        "MODEL": MODEL,
        "API_KEY": API_KEY,
    }
    missing = [
        name for name, value in settings.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Не заданы обязательные настройки LLM: "
            f"{', '.join(missing)}. "
            "Заполните .env в корне проекта "
            "по образцу .env.example."
        )

    if PROVIDER not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "Неподдерживаемый PROVIDER. "
            "Допустимые значения: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )


def complete_messages(messages, tools=None):
    _validate_settings()

    if PROVIDER == "yandex":
        model = f"openai/{MODEL}"
    else:
        model = f"{PROVIDER}/{MODEL}"

    params = {
        "model": model,
        "messages": messages,
        "api_key": API_KEY,
    }

    if PROVIDER == "yandex":
        params["api_base"] = YANDEX_BASE_URL

    if tools:
        params["tools"] = tools

        if PROVIDER == "openai":
            params["reasoning_effort"] = "none"

    return completion(**params)


def ask_messages(messages):
    response = complete_messages(messages)

    return response.choices[0].message.content


def ask(prompt, system_prompt=None):
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    return ask_messages(messages)


if __name__ == "__main__":
    ask("Ответь одним словом: готово")

    providers = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "yandex": "Yandex Cloud",
    }

    model_name = "YandexGPT" if PROVIDER == "yandex" else MODEL
    print(f"\nПривет! Я {model_name} от {providers[PROVIDER]}. Готова к работе.")
