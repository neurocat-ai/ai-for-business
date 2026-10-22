from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infrastructure.llm import ask_messages


PROMPT = """
ВСТАВЬТЕ ПРОМПТ
"""


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
    print("Введите сообщение. Для выхода используйте /exit.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except KeyboardInterrupt:
            print()
            break

        if user_input.lower() == "/exit":
            break

        if not user_input:
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

        print(f"\nИИ: {response}\n")


if __name__ == "__main__":
    main()
