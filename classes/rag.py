from collections import Counter
from pathlib import Path
import json
import math
import re
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infrastructure.llm import ask_messages


ROOT = Path(__file__).resolve().parents[1]
DATASET = "demo"  # demo, training или own

TOP_K = 5
MIN_SCORE = 1.5


RAG_PROMPT = """
ВСТАВЬТЕ ПРОМПТ
"""


SOURCE_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ИСТОЧНИКАМ:
В конце ответа добавь строку USED_SOURCES в формате JSON-массива имен файлов. Указывай только документы, непосредственно подтверждающие конкретные факты ответа. Не включай документ только из-за его присутствия в контексте. Если несколько документов подтверждают один факт, выбери наиболее профильный и не дублируй источники без необходимости. Если документы не использованы, верни USED_SOURCES: [].
"""


STOP_WORDS = {
    "и", "в", "во", "на", "по", "к", "с", "со", "у", "из", "от", "до", "за", "не", "ни",
    "а", "но", "или", "если", "что", "как", "это", "этот", "эта", "для", "при", "бы", "же", "то",
    "вы", "мы", "я", "он", "она", "они", "мне", "вам", "его", "ее", "их", "есть", "был", "была", "были",
    "будет", "быть", "здравствуйте", "привет", "пожалуйста", "нужно", "можно", "хочу",
    "сказать", "подскажите", "делать", "сделать", "получить", "получил",
}

SUFFIXES = sorted({
    "иями", "ями", "ами", "ыми", "ими", "его", "ого", "ему", "ому", "ая", "яя", "ое", "ее",
    "ие", "ые", "ой", "ий", "ый", "ую", "юю", "ем", "им", "ом", "ым", "ах", "ях", "их", "ых", "а", "я", "у", "ю",
    "е", "о", "ы", "и", "ь", "й",
}, key=len, reverse=True)


def clear_terminal():
    print("\033[2J\033[3J\033[H", end="", flush=True)


def normalize_word(word):
    word = word.lower().replace("ё", "е")
    if len(word) <= 4:
        return word

    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def tokenize(text):
    words = re.findall(r"[a-zа-я0-9]+", text.lower().replace("ё", "е"))
    tokens = []

    for word in words:
        if len(word) < 3 or word in STOP_WORDS:
            continue
        normalized = normalize_word(word)
        if normalized not in STOP_WORDS:
            tokens.append(normalized)

    return tokens


def get_knowledge_base_dir(dataset=None):
    selected_dataset = dataset or DATASET
    if selected_dataset not in {"demo", "training", "own"}:
        raise ValueError(f"Неизвестный датасет: {selected_dataset}")
    return ROOT / "data" / selected_dataset / "03_knowledge_base"


def load_knowledge_base(dataset=None):
    knowledge_base_dir = get_knowledge_base_dir(dataset)
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(f"Не найдена база знаний: {knowledge_base_dir}")

    documents = []
    for path in sorted(knowledge_base_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({
            "source": path.name,
            "text": text,
            "tokens": set(tokenize(text)),
        })

    if not documents:
        raise RuntimeError(f"В папке {knowledge_base_dir} нет .md-файлов")
    return documents


def build_document_frequencies(documents):
    frequencies = Counter()
    for document in documents:
        frequencies.update(document["tokens"])
    return frequencies


def retrieve(query, documents, frequencies):
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    document_count = len(documents)
    results = []
    for document in documents:
        score = sum(
            math.log((document_count + 1) / (frequencies[token] + 1)) + 1
            for token in query_tokens & document["tokens"]
        )
        if score >= MIN_SCORE:
            results.append({
                "source": document["source"],
                "text": document["text"],
                "score": score,
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:TOP_K]


def build_search_query(history, user_input):
    previous = [
        message["content"] for message in history
        if message["role"] == "user"
    ]
    return "\n".join(previous[-2:] + [user_input])


def build_knowledge_context(results):
    if not results:
        return "Для текущего сообщения релевантная информация в базе знаний не найдена."
    return "\n\n---\n\n".join(
        f"Источник: {result['source']}\n\n{result['text']}"
        for result in results
    )


def parse_answer_and_sources(content, results):
    content = (content or "Не удалось подготовить ответ.").rstrip()
    source_line = re.search(
        r"(?:^|\n)[ \t]*USED_SOURCES:[ \t]*(.*?)[ \t]*\Z",
        content,
    )
    if not source_line:
        return content.strip(), []

    answer = content[:source_line.start()].strip()
    try:
        parsed_sources = json.loads(source_line.group(1))
    except (json.JSONDecodeError, TypeError):
        parsed_sources = []
    if not isinstance(parsed_sources, list):
        parsed_sources = []

    available_sources = {result["source"] for result in results}
    used_sources = []
    for source in parsed_sources:
        if (
            isinstance(source, str)
            and source in available_sources
            and source not in used_sources
        ):
            used_sources.append(source)
    return answer, used_sources


def main():
    clear_terminal()
    documents = load_knowledge_base()
    frequencies = build_document_frequencies(documents)
    history = []

    print("RAG запущен.")
    print(f"База знаний: {len(documents)} файлов.")
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

        search_query = build_search_query(history, user_input)
        results = retrieve(search_query, documents, frequencies)
        knowledge_context = build_knowledge_context(results)
        system_prompt = (
            f"{RAG_PROMPT}\n\n{SOURCE_INSTRUCTION}\n\n"
            "БАЗА ЗНАНИЙ ДЛЯ ТЕКУЩЕГО СООБЩЕНИЯ:\n\n"
            f"{knowledge_context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]

        response = ask_messages(messages)
        answer, used_sources = parse_answer_and_sources(response, results)
        history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ])

        print(f"\nИИ: {answer}")
        if used_sources:
            print(f"\nИспользованные источники: {', '.join(used_sources)}")
        print()


if __name__ == "__main__":
    main()
