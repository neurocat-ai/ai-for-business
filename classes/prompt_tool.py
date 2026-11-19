from pathlib import Path
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infrastructure.llm import ask_messages


PROMPT = """
ВСТАВЬТЕ ПРОМПТ
"""

SUBMIT_BINDINGS = KeyBindings()


@SUBMIT_BINDINGS.add("enter")
def submit_message(event):
    event.current_buffer.validate_and_handle()


def clear_terminal():
    print("\033[2J\033[3J\033[H", end="", flush=True)


def main():
    clear_terminal()

    messages = [
        {
            "role": "system",
            "content": PROMPT
        }
    ]

    print("Prompt-tool запущен.")
    print(
        "Вставьте сообщение или историю переписки "
        "и нажмите Enter для отправки. "
        "Для выхода используйте /exit.\n"
    )

    session = PromptSession(
        multiline=True,
        key_bindings=SUBMIT_BINDINGS,
    )

    while True:
        try:
            user_input = session.prompt("Вы:\n")
        except KeyboardInterrupt:
            print()
            break

        if user_input.strip().lower() == "/exit":
            break

        if not user_input.strip():
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        response = ask_messages(messages)

        messages.append({
            "role": "assistant",
            "content": response
        })

        print(f"\nИИ:\n{response}\n")


if __name__ == "__main__":
    main()
