from collections import Counter
from pathlib import Path
import json
import math
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infrastructure.llm import ask_messages


ROOT = Path(__file__).resolve().parents[1]
DATASET = "demo"  # demo, training или own
DATASET_TITLES = {
    "demo": "Demo: Клиентская поддержка интернет-магазина",
    "training": "Training: Продажа онлайн-сервиса",
    "own": "Own: Собственный кейс",
}

TOP_K = 5
MIN_SCORE = 1.5
DEBUG = False


def print_startup_header(tool_name, dataset):
    print(f"{tool_name} запущен.")
    print(DATASET_TITLES[dataset])


RAG_PROMPT = """
ВСТАВЬТЕ ПРОМПТ
"""

SUBMIT_BINDINGS = KeyBindings()


@SUBMIT_BINDINGS.add("enter")
def submit_message(event):
    event.current_buffer.validate_and_handle()


SOURCE_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ИСТОЧНИКАМ:
В конце ответа добавь строку USED_SOURCES в формате JSON-массива имен файлов. Указывай только документы, непосредственно подтверждающие конкретные факты ответа. Не включай документ только из-за его присутствия в контексте. Если несколько документов подтверждают один факт, выбери наиболее профильный и не дублируй источники без необходимости. Если документы не использованы, верни USED_SOURCES: [].
"""


SEARCH_DECISION_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ВЫБОРУ ПОИСКА:
Поиск в базе знаний разрешен только в том случае, если RAG_PROMPT содержит явную инструкцию обращаться к базе знаний и условия этой инструкции выполнены для текущего обращения с учетом истории диалога. Общие требования к достоверности, ограничениям ответа или достаточности информации сами по себе не являются разрешением на поиск. Если явной инструкции обращаться к базе знаний нет, верни search=false.

Верни только JSON-объект без markdown:
{"search": true, "query": "краткий самодостаточный поисковый запрос", "sources": ["имя_файла.md"]}
или
{"search": false, "query": "", "sources": []}

Определяй sources на основании RAG_PROMPT, текущего обращения и истории диалога. Указывай точные имена целевых файлов. Если ограничивать поиск конкретными документами не нужно, верни пустой список sources.
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


class EmptyKnowledgeBaseError(RuntimeError):
    def __init__(self, knowledge_base_dir):
        self.knowledge_base_dir = knowledge_base_dir
        super().__init__(f"В папке {knowledge_base_dir} нет .md-файлов")


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
        raise EmptyKnowledgeBaseError(knowledge_base_dir)
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


def parse_search_decision(content, history, user_input):
    content = (content or "").strip()
    try:
        decision = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        json_object = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            decision = json.loads(json_object.group(0)) if json_object else {}
        except (json.JSONDecodeError, TypeError):
            decision = {}

    if not isinstance(decision, dict):
        return False, "", []

    sources = decision.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    target_sources = []
    for source in sources:
        if (
            isinstance(source, str)
            and source.strip()
            and source.strip() not in target_sources
        ):
            target_sources.append(source.strip())

    if decision.get("search") is not True:
        return False, "", target_sources

    query = decision.get("query")
    if not isinstance(query, str) or not query.strip():
        query = build_search_query(history, user_input)
    return True, query.strip(), target_sources


def decide_search(history, user_input):
    messages = [
        {
            "role": "system",
            "content": (
                "RAG_PROMPT:\n"
                f"{RAG_PROMPT}\n\n{SEARCH_DECISION_INSTRUCTION}"
            ),
        },
        *history,
        {"role": "user", "content": user_input},
    ]
    response = ask_messages(messages)
    return parse_search_decision(response, history, user_input)


def select_search_documents(documents, target_sources):
    if not target_sources:
        return documents, []

    available_sources = {document["source"] for document in documents}
    selected_documents = [
        document for document in documents
        if document["source"] in target_sources
    ]
    missing_sources = [
        source for source in target_sources
        if source not in available_sources
    ]
    return selected_documents, missing_sources


def build_knowledge_context(results):
    if not results:
        return "Для текущего сообщения релевантная информация в базе знаний не найдена."
    return "\n\n---\n\n".join(
        f"Источник: {result['source']}\n\n{result['text']}"
        for result in results
    )


def parse_answer_and_reported_sources(content):
    content = (content or "Не удалось подготовить ответ.").rstrip()
    marker = "USED_SOURCES:"
    marker_index = content.rfind(marker)
    if marker_index == -1:
        return content.strip(), []

    answer = content[:marker_index].strip()
    source_block = content[marker_index + len(marker):].strip()
    try:
        parsed_sources = json.loads(source_block)
    except (json.JSONDecodeError, TypeError):
        parsed_sources = []
    if not isinstance(parsed_sources, list):
        parsed_sources = []
    return answer, parsed_sources


def validate_sources(reported_sources, results):
    available_sources = {result["source"] for result in results}
    used_sources = []
    for source in reported_sources:
        if (
            isinstance(source, str)
            and source in available_sources
            and source not in used_sources
        ):
            used_sources.append(source)
    return used_sources


def parse_answer_and_sources(content, results):
    answer, reported_sources = parse_answer_and_reported_sources(content)
    return answer, validate_sources(reported_sources, results)


def print_debug_info(
    search_performed,
    search_query,
    target_sources,
    missing_sources,
    results,
    reported_sources,
    used_sources,
):
    print("\nDEBUG:")
    if search_performed:
        print("Поиск: выполнен")
        print(
            "Поисковый запрос: "
            f"{json.dumps(search_query, ensure_ascii=False)}"
        )
        print(
            "Целевые документы: "
            f"{json.dumps(target_sources, ensure_ascii=False)}"
        )
        if missing_sources:
            print(
                "Целевые документы не найдены: "
                f"{json.dumps(missing_sources, ensure_ascii=False)}"
            )
        found_sources = [result["source"] for result in results]
        print(
            "Найденные документы: "
            f"{json.dumps(found_sources, ensure_ascii=False)}"
        )
    else:
        print("Поиск: не выполнялся")
        print("Поисковый запрос: не сформирован")
        print("Целевые документы: поиск не выполнялся")
        print("Найденные документы: поиск не выполнялся")

    print(
        "Источники, указанные моделью: "
        f"{json.dumps(reported_sources, ensure_ascii=False)}"
    )
    print(
        "Использованные источники после проверки: "
        f"{json.dumps(used_sources, ensure_ascii=False)}"
    )


def main():
    clear_terminal()
    try:
        documents = load_knowledge_base()
    except EmptyKnowledgeBaseError as error:
        print(
            "База знаний пуста. Добавьте хотя бы один .md-файл "
            f"в папку {error.knowledge_base_dir} и повторите запуск."
        )
        return
    frequencies = build_document_frequencies(documents)
    history = []

    print_startup_header("RAG", DATASET)
    print(f"База знаний: {len(documents)} файлов.")
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

        search_performed, search_query, target_sources = decide_search(
            history,
            user_input,
        )
        results = []
        missing_sources = []

        if search_performed:
            search_documents, missing_sources = select_search_documents(
                documents,
                target_sources,
            )
            search_frequencies = frequencies
            if target_sources:
                search_frequencies = build_document_frequencies(
                    search_documents
                )
            results = retrieve(
                search_query,
                search_documents,
                search_frequencies,
            )

        system_prompt_parts = [RAG_PROMPT, SOURCE_INSTRUCTION]
        if search_performed:
            knowledge_context = build_knowledge_context(results)
            system_prompt_parts.append(
                "БАЗА ЗНАНИЙ ДЛЯ ТЕКУЩЕГО СООБЩЕНИЯ:\n\n"
                f"{knowledge_context}"
            )

        messages = [
            {
                "role": "system",
                "content": "\n\n".join(system_prompt_parts),
            },
            *history,
            {"role": "user", "content": user_input},
        ]
        response = ask_messages(messages)
        answer, reported_sources = parse_answer_and_reported_sources(response)
        used_sources = validate_sources(reported_sources, results)

        history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ])

        print(f"\nИИ:\n{answer}")
        if used_sources:
            print(f"\nИспользованные источники: {', '.join(used_sources)}")
        if DEBUG:
            print_debug_info(
                search_performed,
                search_query,
                target_sources,
                missing_sources,
                results,
                reported_sources,
                used_sources,
            )
        print()


if __name__ == "__main__":
    main()
