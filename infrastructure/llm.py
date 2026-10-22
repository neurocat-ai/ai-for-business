from litellm import completion

PROVIDER = "ВСТАВЬТЕ_ПРОВАЙДЕРА"
API_KEY = "ВСТАВЬТЕ_API_КЛЮЧ"
MODEL = "ВСТАВЬТЕ_МОДЕЛЬ"


YANDEX_BASE_URL = "https://ai.api.cloud.yandex.net/v1"


def complete_messages(messages, tools=None):
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
