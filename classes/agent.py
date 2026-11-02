from datetime import datetime
from pathlib import Path
import csv
import json
import re
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from classes.rag import (
    build_document_frequencies,
    load_knowledge_base,
    parse_answer_and_sources,
    retrieve,
)
from infrastructure.llm import complete_messages
from infrastructure.sheets import (
    append_row,
    clear_worksheet,
    get_worksheet,
    initialize_worksheet,
    read_rows,
)


DATASET = "demo"  # demo / training / own
MAX_TOOL_CALLS = 4
SEARCH_TOOL_NAME = "search_knowledge_base"


AGENT_PROMPT = """
ВСТАВЬТЕ ПРОМПТ
"""


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DATE = "2026-10-13"
EXTERNAL_STATE = "04_external_state"

DEMO_ORDERS_FILE = ROOT / "data" / "demo" / EXTERNAL_STATE / "orders.csv"
DEMO_SEED_FILE = ROOT / "data" / "demo" / EXTERNAL_STATE / "return_requests_seed.csv"
TRAINING_LEADS_FILE = ROOT / "data" / "training" / EXTERNAL_STATE / "leads.csv"
TRAINING_SEED_FILE = ROOT / "data" / "training" / EXTERNAL_STATE / "demo_requests_seed.csv"

DEMO_HEADERS = [
    "request_id", "order_id", "item_name", "reason", "comment",
    "created_at", "status",
]
TRAINING_HEADERS = [
    "request_id", "lead_id", "company", "contact", "use_case",
    "preferred_time", "comment", "created_at", "status",
]
OWN_HEADERS = [
    "request_id", "subject", "description", "reference_id", "contact",
    "created_at", "status",
]

TOOL_PROPERTIES = {
    "demo": {
        "order_id": {"type": "string", "description": "Номер заказа"},
        "item": {"type": "string", "description": "Название проблемного товара"},
        "reason": {"type": "string", "description": "Краткая причина обращения"},
        "comment": {"type": "string", "description": "Краткое описание известных фактов"},
    },
    "training": {
        "lead_id": {"type": "string", "description": "ID лида"},
        "company": {"type": "string", "description": "Компания"},
        "contact": {"type": "string", "description": "Контакт для связи"},
        "use_case": {"type": "string", "description": "Конкретный сценарий использования"},
        "preferred_time": {"type": "string", "description": "Пожелание по времени демо"},
        "comment": {"type": "string", "description": "Дополнительный контекст"},
    },
    "own": {
        "subject": {"type": "string", "description": "Тема заявки"},
        "description": {"type": "string", "description": "Описание заявки"},
        "reference_id": {"type": "string", "description": "Внешний идентификатор"},
        "contact": {"type": "string", "description": "Контакт для связи"},
    },
}

CASE_CONFIGS = {
    "demo": {
        "dataset": "demo",
        "title": "Demo: возвраты интернет-магазина",
        "tool_name": "create_return_request",
        "tool_description": "Зарегистрировать заявку на разбор возврата или проблемного заказа.",
        "business_file": DEMO_ORDERS_FILE,
        "business_id": "order_id",
        "seed_file": DEMO_SEED_FILE,
        "headers": DEMO_HEADERS,
        "request_fields": ("order_id", "item", "reason", "comment"),
        "required_fields": ("order_id", "item", "reason", "comment"),
        "optional_fields": (),
        "uppercase_fields": (),
        "not_found_status": "invalid_order",
        "missing_field_in_result": False,
        "request_prefix": "RET",
        "request_start": 1000,
        "active_statuses": {"open"},
        "created_status": "open",
        "id_pattern": r"(?<!\d)(\d{5})(?!\d)",
        "context_missing": "Номер заказа пока не определен.",
        "context_not_found": "Дополнительные данные заказа не найдены.",
        "context_fields": (
            "order_id", "item_name", "category", "price_rub",
            "delivery_date", "delivery_status",
        ),
        "confirmation_title": "Создать заявку?",
        "confirmation_fields": (
            ("order_id", "Заказ"), ("item", "Товар"),
            ("reason", "Причина"), ("comment", "Комментарий"),
        ),
    },
    "training": {
        "dataset": "training",
        "title": "Training: заявки на демо FlowOffice",
        "tool_name": "create_demo_request",
        "tool_description": "Зарегистрировать заявку на проведение демо.",
        "business_file": TRAINING_LEADS_FILE,
        "business_id": "lead_id",
        "seed_file": TRAINING_SEED_FILE,
        "headers": TRAINING_HEADERS,
        "request_fields": (
            "lead_id", "company", "contact", "use_case",
            "preferred_time", "comment",
        ),
        "required_fields": ("lead_id", "company", "contact", "use_case"),
        "optional_fields": ("preferred_time", "comment"),
        "uppercase_fields": ("lead_id",),
        "not_found_status": "lead_not_found",
        "missing_field_in_result": True,
        "request_prefix": "DEMO",
        "request_start": 2000,
        "active_statuses": {"requested", "scheduled"},
        "created_status": "requested",
        "id_pattern": r"(?i)(?<![a-z0-9])LEAD-\d+(?![a-z0-9])",
        "context_missing": "ID лида пока не определен.",
        "context_not_found": "Дополнительные данные лида не найдены.",
        "context_fields": (
            "lead_id", "company", "company_size", "industry",
            "contact_name", "contact", "source", "current_status",
        ),
        "confirmation_title": "Создать заявку на демо?",
        "confirmation_fields": (
            ("lead_id", "Лид"), ("company", "Компания"),
            ("contact", "Контакт"), ("use_case", "Сценарий"),
            ("preferred_time", "Желаемое время"), ("comment", "Комментарий"),
        ),
    },
    "own": {
        "dataset": "own",
        "title": "Own: собственный сценарий",
        "tool_name": "create_own_request",
        "tool_description": "Зарегистрировать заявку собственного бизнес-сценария.",
        "business_file": None,
        "business_id": None,
        "seed_file": None,
        "headers": OWN_HEADERS,
        "request_fields": ("subject", "description", "reference_id", "contact"),
        "required_fields": ("subject", "description"),
        "optional_fields": ("reference_id", "contact"),
        "uppercase_fields": (),
        "missing_field_in_result": True,
        "duplicate_field": "reference_id",
        "request_prefix": "OWN",
        "request_start": 3000,
        "active_statuses": {"requested"},
        "created_status": "requested",
        "confirmation_title": "Создать заявку?",
        "confirmation_fields": (
            ("subject", "Тема"), ("description", "Описание"),
            ("reference_id", "Внешний ID"), ("contact", "Контакт"),
        ),
    },
}


def get_case_config(dataset=None):
    selected_dataset = dataset or DATASET
    try:
        return dict(CASE_CONFIGS[selected_dataset])
    except KeyError as error:
        raise ValueError(f"Неизвестный датасет: {selected_dataset}") from error


def load_csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_business_records(config):
    return {
        str(row[config["business_id"]]): row
        for row in load_csv_rows(config["business_file"])
    }


def load_seed_rows(config):
    if not config.get("seed_file"):
        return []
    return [
        [record.get(header, "") for header in config["headers"]]
        for record in load_csv_rows(config["seed_file"])
    ]


def get_case_worksheet(config):
    return get_worksheet(config["dataset"])


def prepare_case_worksheet(config):
    worksheet = get_case_worksheet(config)
    initialize_worksheet(worksheet, config["headers"], load_seed_rows(config))
    return worksheet


def find_active_duplicate(requests, business_id, business_id_field, active_statuses):
    for request in requests:
        if (
            str(request.get(business_id_field)) == str(business_id)
            and str(request.get("status", "")).lower() in active_statuses
        ):
            return request
    return None


def generate_request_id(requests, prefix, start):
    pattern = re.compile(rf"{re.escape(prefix)}-(\d+)")
    numbers = [
        int(match.group(1))
        for request in requests
        if (match := pattern.fullmatch(str(request.get("request_id", ""))))
    ]
    return f"{prefix}-{max(numbers) + 1 if numbers else start + 1}"


def _normalize_values(config, raw_values):
    values = {}
    for field in config["request_fields"]:
        value = raw_values.get(field)
        if value is None:
            value = ""
        elif field in config["optional_fields"]:
            value = value or ""
        value = str(value).strip()
        if field in config["uppercase_fields"]:
            value = value.upper()
        values[field] = value
    return values


def _missing_result(config, field):
    result = {"status": "missing_required_field"}
    if config["missing_field_in_result"]:
        result["field"] = field
    return result


def _create_request(dataset, raw_values):
    try:
        config = get_case_config(dataset)
        values = _normalize_values(config, raw_values)

        for field in config["required_fields"]:
            if not values[field]:
                return _missing_result(config, field)

        business_id_field = config.get("business_id")
        if business_id_field and config.get("business_file"):
            business_id = values[business_id_field]
            if business_id not in load_business_records(config):
                return {"status": config["not_found_status"]}

        worksheet = prepare_case_worksheet(config)
        requests = read_rows(worksheet)
        duplicate_field = config.get("duplicate_field", business_id_field)
        duplicate_value = values.get(duplicate_field, "")
        duplicate = (
            find_active_duplicate(
                requests,
                duplicate_value,
                duplicate_field,
                config["active_statuses"],
            )
            if duplicate_value else None
        )
        if duplicate:
            return {
                "status": "already_exists",
                "request_id": duplicate["request_id"],
            }

        request_id = generate_request_id(
            requests,
            config["request_prefix"],
            config["request_start"],
        )
        append_row(worksheet, [
            request_id,
            *(values[field] for field in config["request_fields"]),
            datetime.now().isoformat(timespec="seconds"),
            config["created_status"],
        ])
        return {"status": "created", "request_id": request_id}
    except Exception as error:
        return {"status": "tool_error", "error": str(error)}


def create_return_request(order_id, item, reason, comment):
    return _create_request("demo", {
        "order_id": order_id,
        "item": item,
        "reason": reason,
        "comment": comment,
    })


def create_demo_request(
    lead_id,
    company,
    contact,
    use_case,
    preferred_time="",
    comment="",
):
    return _create_request("training", {
        "lead_id": lead_id,
        "company": company,
        "contact": contact,
        "use_case": use_case,
        "preferred_time": preferred_time,
        "comment": comment,
    })


def create_own_request(subject, description, reference_id="", contact=""):
    return _create_request("own", {
        "subject": subject,
        "description": description,
        "reference_id": reference_id,
        "contact": contact,
    })


def reset_requests(dataset=None):
    config = get_case_config(dataset)
    if not config.get("tool_name"):
        return {"status": "tool_not_configured"}

    try:
        initial_rows = load_seed_rows(config)
        worksheet = get_case_worksheet(config)
        clear_worksheet(worksheet)
        initialize_worksheet(worksheet, config["headers"], initial_rows)
        return {"status": "reset"}
    except Exception as error:
        return {"status": "tool_error", "error": str(error)}


def search_knowledge_base(query, documents=None, frequencies=None):
    query = str(query or "").strip()
    if not query:
        return {"query": "", "results": []}

    if documents is None:
        documents = load_knowledge_base(DATASET)
    if frequencies is None:
        frequencies = build_document_frequencies(documents)
    return {
        "query": query,
        "results": retrieve(query, documents, frequencies),
    }


def build_tools(results=None, knowledge_available=False):
    config = get_case_config()
    results = results or []

    properties = {
        name: dict(schema)
        for name, schema in TOOL_PROPERTIES[DATASET].items()
    }
    required = list(config["required_fields"])

    if results:
        properties["used_sources"] = {
            "type": "array",
            "description": "Файлы базы знаний, реально использованные как основание действия.",
            "items": {
                "type": "string",
                "enum": [result["source"] for result in results],
            },
            "minItems": 1,
        }
        required.append("used_sources")

    tools = [{
        "type": "function",
        "function": {
            "name": config["tool_name"],
            "description": config["tool_description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }]
    if knowledge_available:
        tools.append({
            "type": "function",
            "function": {
                "name": SEARCH_TOOL_NAME,
                "description": (
                    "Найти в базе знаний правила и факты, необходимые для ответа "
                    "или решения о выполнении действия."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос к базе знаний.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        })
    return tools


def clear_terminal():
    print("\033[2J\033[3J\033[H", end="", flush=True)


def get_dialog_text(history, user_input):
    messages = [
        message["content"]
        for message in history
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]
    return "\n".join([*messages, user_input])


def _find_context(dataset, history, user_input):
    config = get_case_config(dataset)
    matches = re.findall(config["id_pattern"], get_dialog_text(history, user_input))
    if not matches:
        return config["context_missing"]

    records = load_business_records(config)
    for business_id in reversed(matches):
        business_id = business_id.upper() if dataset == "training" else business_id
        if business_id in records:
            record = records[business_id]
            return "\n".join(
                f"{field}: {record[field]}"
                for field in config["context_fields"]
            )
    return config["context_not_found"]


def find_demo_context(history, user_input):
    return _find_context("demo", history, user_input)


def find_training_context(history, user_input):
    return _find_context("training", history, user_input)


def find_business_context(history, user_input):
    if DATASET == "demo":
        return find_demo_context(history, user_input)
    if DATASET == "training":
        return find_training_context(history, user_input)
    return "Бизнес-контекст не настроен."


def build_system_prompt(business_context):
    scenario_date = (
        ""
        if DATASET == "own"
        else f"Дата учебного сценария: {SCENARIO_DATE}\n\n"
    )
    return (
        f"{AGENT_PROMPT}\n\n"
        f"{scenario_date}"
        f"БИЗНЕС-ДАННЫЕ:\n\n{business_context}"
    )


def get_tool_call(message):
    tool_calls = getattr(message, "tool_calls", None)
    return tool_calls[0] if tool_calls else None


def parse_tool_call(tool_call):
    name, arguments = _parse_function_call(tool_call)
    expected_name = get_case_config()["tool_name"]
    if name != expected_name:
        raise RuntimeError(f"Неизвестный инструмент: {name}")
    return arguments


def _parse_function_call(tool_call):
    function = tool_call.function
    arguments = json.loads(function.arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Аргументы инструмента должны быть объектом.")
    return function.name, arguments


def _allowed_sources(sources, results):
    available = {result["source"] for result in results}
    allowed = []
    for source in sources:
        if isinstance(source, str) and source in available and source not in allowed:
            allowed.append(source)
    return allowed


def validate_tool_arguments(arguments, results=None, knowledge_used=False):
    if not isinstance(arguments, dict):
        raise ValueError("Аргументы инструмента должны быть объектом.")

    config = get_case_config()
    for field in config["required_fields"]:
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Не заполнено обязательное поле: {field}.")

    for field in config["optional_fields"]:
        arguments[field] = str(arguments.get(field) or "").strip()

    if not knowledge_used:
        arguments.pop("used_sources", None)
        return arguments

    raw_sources = arguments.get("used_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("Не заполнено обязательное поле: used_sources.")

    arguments["used_sources"] = _allowed_sources(raw_sources, results)
    if not arguments["used_sources"]:
        raise ValueError("Не указано допустимое основание действия.")
    return arguments


def make_confirmation(arguments):
    config = get_case_config()
    lines = [config["confirmation_title"]]
    for field, label in config["confirmation_fields"]:
        if arguments.get(field):
            lines.append(f"{label}: {arguments[field]}")
    if arguments.get("used_sources"):
        lines.append("Основание: " + ", ".join(arguments["used_sources"]))
    return "\n".join([*lines, "", "Ответьте да или нет."])


def execute_tool(arguments):
    if DATASET == "demo":
        return create_return_request(
            arguments["order_id"],
            arguments["item"],
            arguments["reason"],
            arguments["comment"],
        )
    if DATASET == "training":
        return create_demo_request(
            arguments["lead_id"],
            arguments["company"],
            arguments["contact"],
            arguments["use_case"],
            arguments.get("preferred_time", ""),
            arguments.get("comment", ""),
        )
    return create_own_request(
        arguments["subject"],
        arguments["description"],
        arguments.get("reference_id", ""),
        arguments.get("contact", ""),
    )


def build_tool_result_answer(tool_result):
    status = tool_result.get("status")

    if DATASET == "own":
        if status == "created":
            return (
                "Заявка зарегистрирована. "
                f"Номер заявки: {tool_result['request_id']}."
            )
        if status == "already_exists":
            return (
                "По этому внешнему ID уже есть заявка "
                f"{tool_result['request_id']}. Новую заявку я не создавал."
            )
        if status == "missing_required_field":
            suffix = (
                f" Не заполнено поле: {tool_result['field']}."
                if tool_result.get("field") else ""
            )
            return "Не удалось зарегистрировать заявку, потому что не хватает обязательных данных." + suffix
        return "Не удалось зарегистрировать заявку из-за технической ошибки."

    if DATASET == "training":
        if status == "created":
            return (
                "Заявка на демо зарегистрирована. "
                f"Номер заявки: {tool_result['request_id']}. "
                "Это не означает, что время демо уже подтверждено."
            )
        if status == "already_exists":
            return (
                "По этому лиду уже есть заявка на демо "
                f"{tool_result['request_id']}. Новую заявку я не создавал."
            )
        if status == "lead_not_found":
            return (
                "Не удалось зарегистрировать заявку: лид с таким ID не найден. "
                "Проверьте, пожалуйста, ID лида."
            )
        if status == "missing_required_field":
            suffix = (
                f" Не заполнено поле: {tool_result['field']}."
                if tool_result.get("field") else ""
            )
            return "Не удалось зарегистрировать заявку, потому что не хватает обязательных данных." + suffix
        return "Не удалось зарегистрировать заявку на демо из-за технической ошибки."

    if status == "created":
        return (
            "Заявка на разбор зарегистрирована. "
            f"Номер заявки: {tool_result['request_id']}. "
            "Это не означает автоматического одобрения возврата или перечисления денег."
        )
    if status == "already_exists":
        return (
            f"По этому случаю уже есть заявка {tool_result['request_id']}. "
            "Новую заявку я не создавал. Этот случай нужно передать специалисту "
            "для дальнейшего разбора."
        )
    if status == "invalid_order":
        return (
            "Не удалось зарегистрировать заявку: заказ с таким номером не найден. "
            "Проверьте, пожалуйста, номер заказа."
        )
    if status == "missing_required_field":
        return "Не удалось зарегистрировать заявку, потому что не хватает обязательных данных."
    return (
        "Не удалось зарегистрировать заявку из-за технической ошибки. "
        "Этот случай нужно передать специалисту."
    )


def _matches_phrase(text, phrases):
    normalized = text.lower().strip().replace("ё", "е")
    return any(
        normalized == phrase or normalized.startswith(phrase + " ")
        for phrase in phrases
    )


def is_yes(text):
    return _matches_phrase(text, (
        "да", "подтверждаю", "создавай", "создавайте", "оформляй", "оформляйте",
    ))


def is_no(text):
    return _matches_phrase(text, ("нет", "не надо", "отмена", "отменить"))


def _append_exchange(history, user_input, answer):
    history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ])


def _print_sources(sources):
    if sources:
        print("\nОснование: " + ", ".join(sources))


def _handle_pending(user_input, pending, history, config):
    if is_no(user_input):
        answer = "Хорошо, действие отменено."
        _append_exchange(history, user_input, answer)
        print(f"\nИИ: {answer}\n")
        return None

    if not is_yes(user_input):
        print(
            "\nИИ: Перед выполнением действия нужно подтверждение. "
            "Ответьте да или нет.\n"
        )
        return pending

    arguments = pending["arguments"]
    tool_result = execute_tool(arguments)
    answer = build_tool_result_answer(tool_result)
    _append_exchange(history, user_input, answer)
    print(f"\nИнструмент: {config['tool_name']}")
    print(f"\nИИ: {answer}")
    _print_sources(arguments.get("used_sources", []))
    print()
    return None


def _load_optional_knowledge_base():
    try:
        documents = load_knowledge_base(DATASET)
    except (FileNotFoundError, RuntimeError):
        return [], {}
    return documents, build_document_frequencies(documents)


def _merge_results(current, new):
    merged = list(current)
    known_sources = {result["source"] for result in current}
    for result in new:
        if result["source"] not in known_sources:
            merged.append(result)
            known_sources.add(result["source"])
    return merged


def _print_search_call():
    print(f"Инструмент: {SEARCH_TOOL_NAME}")


def _append_tool_result(messages, message, tool_call, result, call_number):
    call_id = getattr(tool_call, "id", None) or f"call_{call_number}"
    messages.extend([
        {
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": tool_call.function.name,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ])


def run_agent_turn(history, user_input, documents, frequencies):
    business_context = find_business_context(history, user_input)
    messages = [
        {"role": "system", "content": build_system_prompt(business_context)},
        *history,
        {"role": "user", "content": user_input},
    ]
    knowledge_available = bool(documents)
    search_attempted = False
    results = []

    for call_number in range(1, MAX_TOOL_CALLS + 1):
        response = complete_messages(
            messages,
            tools=build_tools(results, knowledge_available),
        )
        message = response.choices[0].message
        tool_call = get_tool_call(message)
        if not tool_call:
            answer, used_sources = parse_answer_and_sources(
                message.content,
                results,
            )
            return {
                "type": "answer",
                "answer": answer,
                "used_sources": used_sources,
            }

        try:
            tool_name, arguments = _parse_function_call(tool_call)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            result = {"status": "invalid_arguments", "error": str(error)}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        if tool_name == SEARCH_TOOL_NAME:
            search_attempted = True
            search_result = search_knowledge_base(
                arguments.get("query", ""),
                documents,
                frequencies,
            )
            _print_search_call()
            results = _merge_results(results, search_result["results"])
            _append_tool_result(
                messages, message, tool_call, search_result, call_number,
            )
            continue

        if tool_name != get_case_config()["tool_name"]:
            result = {"status": "unknown_tool", "tool": tool_name}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        if search_attempted and not results:
            result = {
                "status": "insufficient_knowledge",
                "message": (
                    "База знаний не дала основания для выполнения действия. "
                    "Действие не выполнено."
                ),
            }
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        try:
            validated = validate_tool_arguments(
                arguments,
                results,
                knowledge_used=search_attempted,
            )
        except ValueError as error:
            result = {"status": "invalid_arguments", "error": str(error)}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue
        return {"type": "action", "arguments": validated}

    return {
        "type": "answer",
        "answer": (
            "Не удалось завершить обработку запроса: превышен лимит "
            "последовательных вызовов инструментов."
        ),
        "used_sources": [],
    }


def main():
    clear_terminal()
    config = get_case_config()
    documents, frequencies = _load_optional_knowledge_base()

    history = []
    pending = None
    print("Agent запущен.")
    print(f"Кейс: {config['title']}.")
    if documents:
        print(f"База знаний: {len(documents)} файлов.")
    else:
        print("База знаний: не подключена.")
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
        if pending:
            pending = _handle_pending(user_input, pending, history, config)
            continue

        turn = run_agent_turn(
            history,
            user_input,
            documents,
            frequencies,
        )
        if turn["type"] == "answer":
            answer = turn["answer"]
            used_sources = turn["used_sources"]
            _append_exchange(history, user_input, answer)
            print(f"\nИИ: {answer}")
            _print_sources(used_sources)
            print()
            continue

        arguments = turn["arguments"]
        confirmation = make_confirmation(arguments)
        _append_exchange(history, user_input, confirmation)
        pending = {"arguments": arguments}
        print(f"\nИИ: {confirmation}\n")


if __name__ == "__main__":
    main()
