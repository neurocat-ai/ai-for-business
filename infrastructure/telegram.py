import json
import os
import socket
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import truststore


ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT = 10
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class TelegramAPIError(RuntimeError):
    def __init__(self, message, status_code=None, temporary=False):
        super().__init__(message)
        self.status_code = status_code
        self.temporary = temporary


def _settings(require_chat_id=True):
    load_dotenv(ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    missing = []
    if not token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if require_chat_id and not chat_id:
        missing.append("TELEGRAM_CHAT_ID")
    return token, chat_id, missing


def _safe_error(message, token):
    text = str(message or "Неизвестная ошибка Telegram.")
    return text.replace(token, "[скрыто]") if token else text


def _request(method, token, payload=None, request_timeout=REQUEST_TIMEOUT):
    data = urlencode(payload or {}).encode("utf-8")
    request = Request(
        f"{API_BASE}/bot{token}/{method}",
        data=data,
        method="POST",
    )
    try:
        with urlopen(
            request,
            timeout=request_timeout,
            context=SSL_CONTEXT,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            result = json.loads(error.read().decode("utf-8"))
            description = result.get("description")
        except (json.JSONDecodeError, UnicodeDecodeError):
            description = None
        raise TelegramAPIError(
            description or f"Telegram API вернул HTTP {error.code}.",
            status_code=error.code,
            temporary=error.code >= 500,
        ) from error
    except (TimeoutError, socket.timeout) as error:
        raise TelegramAPIError(
            "Превышено время ожидания ответа Telegram.",
            temporary=True,
        ) from error
    except URLError as error:
        raise TelegramAPIError(
            "Не удалось подключиться к Telegram API.",
            temporary=True,
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise TelegramAPIError(
            "Telegram API вернул некорректный ответ."
        ) from error

    if not isinstance(result, dict) or result.get("ok") is not True:
        description = (
            result.get("description")
            if isinstance(result, dict) else None
        )
        status_code = (
            result.get("error_code")
            if isinstance(result, dict) else None
        )
        raise TelegramAPIError(
            description or "Telegram API не подтвердил выполнение запроса.",
            status_code=status_code,
            temporary=(
                isinstance(status_code, int) and status_code >= 500
            ),
        )
    return result.get("result")


def send_notification(text):
    token, chat_id, missing = _settings(require_chat_id=True)
    if missing:
        return {
            "status": "configuration_error",
            "missing": missing,
            "error": (
                "Не заданы настройки Telegram: " + ", ".join(missing) + "."
            ),
        }

    protected_values = [
        token,
        os.getenv("API_KEY", "").strip(),
    ]
    if not isinstance(text, str) or not text.strip():
        return {
            "status": "send_error",
            "error": "Текст уведомления пуст.",
        }
    if any(value and value in text for value in protected_values):
        return {
            "status": "send_error",
            "error": "Уведомление содержит защищенные данные и не отправлено.",
        }

    try:
        result = _request("sendMessage", token, {
            "chat_id": chat_id,
            "text": text,
        })
    except TelegramAPIError as error:
        return {
            "status": "send_error",
            "error": _safe_error(error, token),
        }

    return {
        "status": "sent",
        "message_id": result.get("message_id") if isinstance(result, dict) else None,
    }


def get_available_chats(timeout=0, offset=None):
    token, _, missing = _settings(require_chat_id=False)
    if missing:
        return {
            "status": "configuration_error",
            "missing": missing,
            "error": "Не задан TELEGRAM_BOT_TOKEN.",
        }

    payload = {"timeout": timeout}
    if offset is not None:
        payload["offset"] = offset

    try:
        updates = _request(
            "getUpdates",
            token,
            payload,
            request_timeout=max(REQUEST_TIMEOUT, timeout + 5),
        )
    except TelegramAPIError as error:
        return {
            "status": "request_error",
            "error": _safe_error(error, token),
            "status_code": error.status_code,
            "temporary": error.temporary,
        }

    chats = {}
    update_ids = []
    for update in updates if isinstance(updates, list) else []:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            update_ids.append(update_id)
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is not None:
            chats[str(chat_id)] = {
                "id": chat_id,
                "type": chat.get("type", "unknown"),
                "name": (
                    chat.get("title")
                    or " ".join(filter(None, [
                        chat.get("first_name"),
                        chat.get("last_name"),
                    ]))
                    or chat.get("username")
                    or "Без названия"
                ),
            }
    next_offset = max(update_ids) + 1 if update_ids else offset
    return {
        "status": "ok",
        "chats": list(chats.values()),
        "next_offset": next_offset,
    }


def _print_chats(chats):
    if len(chats) == 1:
        chat = chats[0]
        print(f"Найден чат: {chat['name']} ({chat['type']})")
        print(f"TELEGRAM_CHAT_ID={chat['id']}")
        return

    print("Найдено несколько чатов. Выберите нужный и скопируйте его ID в .env:")
    for chat in chats:
        print(f"- {chat['name']} ({chat['type']}): TELEGRAM_CHAT_ID={chat['id']}")


def main():
    token, _, missing = _settings(require_chat_id=False)
    if missing:
        print(
            "Добавьте TELEGRAM_BOT_TOKEN в .env в корне проекта, затем "
            "повторите запуск. Chat ID на этом этапе не требуется."
        )
        return

    try:
        offset = None
        waiting_started = False
        while True:
            timeout = 20 if waiting_started else 0
            result = get_available_chats(timeout=timeout, offset=offset)
            if result["status"] != "ok":
                if result.get("status_code") == 401:
                    print(
                        "Ошибка авторизации Telegram. Проверьте "
                        "TELEGRAM_BOT_TOKEN."
                    )
                    return
                if result.get("status_code") == 409:
                    print(
                        "Конфликт получения обновлений. Остановите другой "
                        "запущенный экземпляр бота и повторите запуск."
                    )
                    return
                if result.get("temporary"):
                    print("Временная ошибка соединения. Повтор через 2 секунды.")
                    time.sleep(2)
                    continue
                print(f"Не удалось получить чаты: {result['error']}")
                return

            offset = result.get("next_offset", offset)
            if not waiting_started:
                print("Откройте своего Telegram-бота и отправьте ему команду /start.")
                print("Ожидание нового сообщения... Для выхода нажмите Ctrl+C.")
                waiting_started = True
                continue

            chats = result["chats"]
            if chats:
                _print_chats(chats)
                return
    except KeyboardInterrupt:
        print("\nПолучение Chat ID остановлено.")


if __name__ == "__main__":
    main()
