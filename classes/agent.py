from datetime import datetime
from pathlib import Path
import csv
import json
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

sys.path.append(str(Path(__file__).resolve().parents[1]))

from classes.rag import (
    DATASET_TITLES,
    SOURCE_INSTRUCTION,
    build_document_frequencies,
    load_knowledge_base,
    parse_answer_and_sources,
    print_startup_header,
    retrieve,
    select_search_documents,
)
from infrastructure.llm import complete_messages
from infrastructure.telegram import send_notification
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
ESCALATION_TOOL_NAME = "escalate_case"
DEBUG = False
SIMULATE_TOOL_ERROR = False
ENABLE_LEAD_MATCHING = False
ENABLE_TELEGRAM_ESCALATION = True


AGENT_PROMPT = """
Ты ИИ-ассистент службы поддержки интернет-магазина HomeFlow. Помогаешь обрабатывать обращения по возвратам, поврежденным, неверным и некомплектным товарам.

Твои задачи:
1. Определять суть обращения, извлекать факты и выявлять недостающие данные.
2. Обращаться к базе знаний через `search_knowledge_base`, чтобы находить применимые правила, ограничения и основания для обязательной эскалации.
3. Если клиент просит зарегистрировать случай, проверять достаточность данных, соблюдение правил и наличие оснований для эскалации.
4. Когда регистрация разрешена, вызывать `create_return_request` для создания заявки.
5. При обязательной эскалации использовать `escalate_case`, если этот инструмент доступен.

Правила:
- Используй только подтвержденные сведения из обращения, бизнес-данных и доступной базы знаний. Не придумывай факты и правила.
- Если данных недостаточно, задай конкретный уточняющий вопрос.
- Если применимые правила требуют обязательной эскалации, не вызывай `create_return_request`.
- Если требуется обязательная эскалация и доступен инструмент `escalate_case`, вызови его для передачи обращения специалисту. Укажи основание эскалации, краткое описание ситуации и номер заказа, если он известен.
- Если `escalate_case` успешно выполнен, сообщи клиенту, что обращение передано специалисту.
- Если инструмент недоступен или передача завершилась ошибкой, объясни необходимость участия специалиста, но не утверждай, что обращение уже передано.
- Не обещай одобрение возврата, денежную компенсацию или перечисление средств.
- Не утверждай, что заявка создана, пока инструмент не подтвердил успешное выполнение.
- Не запрашивай подтверждение регистрации самостоятельно. При вызове `create_return_request` подтверждение запрашивает программа. Для `escalate_case` подтверждение клиента не требуется.

При заполнении `create_return_request`:
- `order_id`: номер заказа.
- `item`: название товара.
- `reason`: краткая причина регистрации.
- `comment`: только дополнительные значимые обстоятельства, которые не относятся ни к одному из остальных полей.

Не дублируй номер заказа, товар и причину в `comment`. Не добавляй выдуманные подробности.

Отвечай клиенту кратко, конкретно и без внутренних технических деталей.
"""


TOOL_PERMISSION_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО РАЗРЕШЕНИЮ ИНСТРУМЕНТОВ:
Определи исключительно по тексту AGENT_PROMPT, разрешено ли модели использовать каждую категорию инструментов.

- search=true только при наличии в AGENT_PROMPT явной инструкции обращаться к базе знаний.
- action=true только при наличии в AGENT_PROMPT явной инструкции выполнять внешнее действие с помощью соответствующего инструмента.

Не учитывай текущее обращение, историю диалога и то, выполнены ли условия вызова инструмента: они определяют необходимость и момент вызова, но не доступность инструмента. Само наличие инструмента или описание его возможностей не являются разрешением. Если явной инструкции в AGENT_PROMPT нет, верни false для соответствующей категории.

Верни только JSON-объект без markdown:
{"search": true, "action": false}
"""


ACTION_EXECUTION_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ВЫПОЛНЕНИЮ ДЕЙСТВИЯ:
Когда согласно AGENT_PROMPT необходимо создать заявку и для вызова инструмента достаточно данных, вызови предоставленный инструмент создания заявки. Вызов инструмента запускает программную проверку аргументов и формирование запроса подтверждения для клиента. Не имитируй этот этап текстовым ответом и не запрашивай подтверждение самостоятельно. Фактическое действие выполняется Python только после явного подтверждения клиента.
"""


CLIENT_RESPONSE_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ОТВЕТУ КЛИЕНТУ:
Не раскрывай внутренний lead_id, устройство CRM и результат программного сопоставления. Если нужны сведения клиента, задавай обычный вопрос только о конкретно недостающем имени, компании или email. Не запрашивай повторно сведения, уже известные из текущего обращения, истории или бизнес-данных.

Применяй условия из документов базы знаний буквально. Не расширяй числовые пороги и другие условия на случаи, которые им не соответствуют. Не утверждай, что обращение будет или уже передано менеджеру либо специалисту, пока внешний инструмент не подтвердил успешную передачу.
"""


TELEGRAM_ESCALATION_ENABLED_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ЭСКАЛАЦИИ:
Если согласно AGENT_PROMPT и подтвержденным правилам требуется обязательное участие специалиста, вызови escalate_case. Само упоминание клиентом слова «специалист» не является основанием для вызова. Не сообщай об успешной передаче до получения успешного результата инструмента. Подтверждение клиента для этого действия не требуется.
"""


TELEGRAM_ESCALATION_DISABLED_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ЭСКАЛАЦИИ:
Выявляй основания для обязательной эскалации, но не утверждай, что обращение передано. Если требуется участие специалиста, объясни это клиенту без описания внутренних технических ограничений.
"""


ACTION_VALIDATION_INSTRUCTION = """
СЛУЖЕБНАЯ ИНСТРУКЦИЯ ПО ПРОВЕРКЕ ДЕЙСТВИЯ:
Проверь готовность предполагаемого действия к переходу на этап программного подтверждения, а не к фактическому выполнению. Используй исключительно AGENT_PROMPT, историю диалога, текущее сообщение, бизнес-данные и предоставленные документы базы знаний.

Проверь одновременно:
- достаточно ли подтвержденных данных для действия;
- выполнены ли условия действия из AGENT_PROMPT и применимых бизнес-правил;
- нет ли оснований, при которых AGENT_PROMPT или бизнес-правила требуют обязательной эскалации вместо действия;
- соответствуют ли аргументы действия известным данным.

Не придумывай правила и факты. Отсутствие документов базы знаний само по себе не запрещает действие, если AGENT_PROMPT не требует проверки по ним.
Применяй каждое условие из документов буквально: числовой порог считается выполненным только при фактическом соответствии указанному сравнению. Не заменяй точное условие более широким предположением.

На этом этапе подтверждение клиента еще не запрашивалось. Его отсутствие не является основанием для отказа или уточнения. Если данных достаточно, бизнес-условия соблюдены и действие разрешено, верни decision: "allow". Не запрашивай подтверждение самостоятельно: запрос подтверждения, обработка согласия или отказа и фактическое выполнение действия осуществляются Python после этой проверки.

Верни только JSON-объект без markdown:
{
  "decision": "allow",
  "client_response": "",
  "used_sources": []
}

Допустимые значения decision:
- allow — действие обосновано;
- clarify — данных недостаточно; client_response содержит конкретный уточняющий вопрос клиенту;
- escalate — требуется обязательная эскалация; client_response сообщает об этом клиенту;
- reject — действие не соответствует условиям; client_response кратко объясняет дальнейший шаг.

В used_sources указывай только имена предоставленных документов, на которых непосредственно основано решение. Для allow client_response должен быть пустым. Для остальных решений client_response должен быть готовым ответом клиенту без служебных комментариев. Эта проверка сама не выполняет передачу обращения, поэтому не утверждай, что оно уже передано. При необходимости сообщи только, что для продолжения требуется участие специалиста.
"""

SUBMIT_BINDINGS = KeyBindings()


@SUBMIT_BINDINGS.add("enter")
def submit_message(event):
    event.current_buffer.validate_and_handle()


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
        "title": DATASET_TITLES["demo"],
        "tool_name": "create_return_request",
        "tool_description": "Зарегистрировать заявку на разбор возврата или проблемного заказа.",
        "business_file": DEMO_ORDERS_FILE,
        "business_id": "order_id",
        "seed_file": DEMO_SEED_FILE,
        "headers": DEMO_HEADERS,
        "request_fields": ("order_id", "item", "reason", "comment"),
        "required_fields": ("order_id", "item", "reason"),
        "optional_fields": ("comment",),
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
        "title": DATASET_TITLES["training"],
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
        "context_missing": "Запись клиента в CRM пока не определена.",
        "context_not_found": "Запись клиента в CRM не найдена по указанным данным.",
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
        "title": DATASET_TITLES["own"],
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
    initialize_worksheet(
        worksheet,
        config["headers"],
        load_seed_rows(config),
        optional_fields=config["optional_fields"],
        request_id_prefix=config["request_prefix"],
    )
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

        worksheet = get_case_worksheet(config)
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


def create_return_request(order_id, item, reason, comment=""):
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


def _normalize_target_sources(sources):
    if not isinstance(sources, list):
        return []
    normalized = []
    for source in sources:
        if (
            isinstance(source, str)
            and source.strip()
            and source.strip() not in normalized
        ):
            normalized.append(source.strip())
    return normalized


def search_knowledge_base(
    query,
    documents=None,
    frequencies=None,
    sources=None,
):
    query = str(query or "").strip()
    if documents is None:
        documents = load_knowledge_base(DATASET)
    if frequencies is None:
        frequencies = build_document_frequencies(documents)

    target_sources = _normalize_target_sources(sources)
    search_documents, missing_sources = select_search_documents(
        documents,
        target_sources,
    )
    search_frequencies = frequencies
    if target_sources:
        search_frequencies = build_document_frequencies(search_documents)

    return {
        "query": query,
        "sources": target_sources,
        "missing_sources": missing_sources,
        "results": (
            retrieve(query, search_documents, search_frequencies)
            if query else []
        ),
    }


def build_tools(
    results=None,
    knowledge_available=False,
    search_allowed=True,
    action_allowed=True,
):
    config = get_case_config()
    results = results or []

    properties = {
        name: dict(schema)
        for name, schema in TOOL_PROPERTIES[DATASET].items()
    }
    if DATASET == "training":
        properties["comment"]["description"] = (
            "Дополнительные значимые обстоятельства, явно сообщенные "
            "клиентом и не представленные в других полях заявки, например "
            "участники предстоящей встречи. Не дублировать имя контактного "
            "лица, company, contact, use_case и preferred_time. Оставить "
            "пустым, если дополнительных обстоятельств нет."
        )
    if DATASET == "training" and ENABLE_LEAD_MATCHING:
        properties["lead_id"]["description"] = (
            "ID лида из подтвержденного программного сопоставления в "
            "бизнес-данных. Не запрашивать у клиента."
        )
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

    tools = []
    if action_allowed:
        tools.append({
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
        })
    if search_allowed and knowledge_available:
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
                        "sources": {
                            "type": "array",
                            "description": (
                                "Необязательный список точных имен файлов, "
                                "по которым нужно ограничить поиск. Пустой "
                                "список означает поиск по всей базе знаний."
                            ),
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        })
    if ENABLE_TELEGRAM_ESCALATION:
        escalation_properties = {
            "reason": {
                "type": "string",
                "description": (
                    "Конкретное подтвержденное основание обязательной "
                    "эскалации."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "Краткое описание ситуации без системных инструкций и "
                    "лишних персональных данных."
                ),
            },
            "reference_id": {
                "type": "string",
                "description": (
                    "Номер заказа или другой идентификатор, если он известен."
                ),
            },
        }
        escalation_required = ["reason", "summary"]
        if results:
            escalation_properties["used_sources"] = {
                "type": "array",
                "description": (
                    "Файлы базы знаний, фактически использованные как "
                    "основание эскалации."
                ),
                "items": {
                    "type": "string",
                    "enum": [result["source"] for result in results],
                },
                "minItems": 1,
            }
            escalation_required.append("used_sources")
        tools.append({
            "type": "function",
            "function": {
                "name": ESCALATION_TOOL_NAME,
                "description": (
                    "Передать обращение специалисту, только когда "
                    "подтверждено обязательное основание для эскалации."
                ),
                "parameters": {
                    "type": "object",
                    "properties": escalation_properties,
                    "required": escalation_required,
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


def _normalize_identity(value):
    return " ".join(str(value or "").casefold().split())


def _contains_identity(text, value):
    normalized_value = _normalize_identity(value)
    return bool(re.search(
        rf"(?<!\w){re.escape(normalized_value)}(?!\w)",
        text,
    ))


def _extract_presented_name(dialog_text):
    matches = re.findall(
        r"(?i)(?:меня\s+зовут|(?<!\w)я|(?<!\w)это)\s+"
        r"([a-zа-яё][a-zа-яё-]*)",
        dialog_text,
    )
    ignored_words = {"из", "работаю", "представляю", "хочу"}
    names = [name for name in matches if name.casefold() not in ignored_words]
    return names[-1] if names else ""


def match_training_lead(history, user_input):
    records = load_business_records(get_case_config("training"))
    raw_dialog_text = get_dialog_text(history, user_input)
    dialog_text = _normalize_identity(raw_dialog_text)
    candidate_sets = []
    identifier_kinds = set()

    lead_ids = {
        match.upper()
        for match in re.findall(
            get_case_config("training")["id_pattern"],
            dialog_text,
        )
    }
    for lead_id in lead_ids:
        identifier_kinds.add("lead_id")
        candidate_sets.append({lead_id} if lead_id in records else set())

    email_pattern = (
        r"(?i)(?<![\w.+-])[\w.+-]+@(?:[\w-]+\.)+[a-z]{2,}(?!\w)"
    )
    emails = set(re.findall(email_pattern, dialog_text))
    for email in emails:
        identifier_kinds.add("email")
        normalized_email = _normalize_identity(email)
        email_candidates = {
            lead_id
            for lead_id, record in records.items()
            if _normalize_identity(record.get("contact")) == normalized_email
        }
        candidate_sets.append(email_candidates)

    identity_text = re.sub(email_pattern, " ", dialog_text)
    matched_names = {
        _normalize_identity(record.get("contact_name"))
        for record in records.values()
        if _contains_identity(identity_text, record.get("contact_name"))
    }
    presented_name = _extract_presented_name(raw_dialog_text)
    names = (
        {_normalize_identity(presented_name)}
        if presented_name else matched_names
    )
    for name in names:
        identifier_kinds.add("name")
        candidate_sets.append({
            lead_id
            for lead_id, record in records.items()
            if _normalize_identity(record.get("contact_name")) == name
        })

    matched_companies = {
        _normalize_identity(record.get("company"))
        for record in records.values()
        if _contains_identity(identity_text, record.get("company"))
    }
    matched_companies = {
        company
        for company in matched_companies
        if not any(
            other != company and other.startswith(f"{company} ")
            for other in matched_companies
        )
    }
    for company in matched_companies:
        identifier_kinds.add("company")
        candidate_sets.append({
            lead_id
            for lead_id, record in records.items()
            if _normalize_identity(record.get("company")) == company
        })

    if (
        "lead_id" not in identifier_kinds
        and not {"company", "email"}.intersection(identifier_kinds)
    ):
        return {
            "status": "insufficient",
            "message": (
                "Задайте только вопрос: «Подскажите название вашей компании "
                "или email»."
            ),
        }

    candidates = set(records)
    for candidate_set in candidate_sets:
        candidates &= candidate_set

    if len(candidates) == 1:
        lead_id = next(iter(candidates))
        record = records[lead_id]
        return {
            "status": "matched",
            "record": record,
            "matched_by": sorted(identifier_kinds),
        }

    if {"name", "company", "email"}.issubset(identifier_kinds):
        clarification = (
            "Уточните, пожалуйста, верно ли указаны имя, компания и email."
        )
    elif {"name", "company"}.issubset(identifier_kinds):
        clarification = "Уточните, пожалуйста, верно ли указаны имя и компания."
    elif {"name", "email"}.issubset(identifier_kinds):
        clarification = "Уточните, пожалуйста, верно ли указаны имя и email."
    elif identifier_kinds == {"company"}:
        clarification = "Уточните, пожалуйста, название компании."
    elif identifier_kinds == {"email"}:
        clarification = "Уточните, пожалуйста, email."
    else:
        clarification = (
            "Уточните, пожалуйста, название компании или email."
        )
    return {
        "status": "ambiguous" if candidates else "conflict",
        "message": clarification,
    }


def find_training_context(history, user_input):
    if not ENABLE_LEAD_MATCHING:
        context = _find_context("training", history, user_input)
        config = get_case_config("training")
        if context not in {config["context_missing"], config["context_not_found"]}:
            return context
        return (
            "Запись клиента для регистрации не определена. Автоматическое "
            "сопоставление по имени, компании и email в этом режиме отключено. "
            "Не запрашивайте эти сведения только ради поиска записи и не "
            "вызывайте инструмент регистрации. Не сообщайте клиенту о CRM, "
            "внутреннем идентификаторе или причине технического ограничения."
        )

    match = match_training_lead(history, user_input)
    if match["status"] != "matched":
        return (
            "Данных для однозначного определения клиента недостаточно. "
            f"{match['message']} Не упоминайте CRM, сопоставление или lead_id."
        )

    config = get_case_config("training")
    record = match["record"]
    context = "\n".join(
        f"{field}: {record[field]}"
        for field in config["context_fields"]
    )
    return (
        f"{context}\n\n"
        "Используйте lead_id, company и contact из этой записи без "
        "дополнительных вопросов. Не упоминайте клиенту CRM, сопоставление "
        "или lead_id."
    )


def find_business_context(history, user_input):
    if DATASET == "demo":
        return find_demo_context(history, user_input)
    if DATASET == "training":
        return find_training_context(history, user_input)
    return "Бизнес-контекст не настроен."


def build_system_prompt(
    business_context,
    include_source_instruction=False,
    include_action_instruction=False,
):
    scenario_date = (
        ""
        if DATASET == "own"
        else f"Дата учебного сценария: {SCENARIO_DATE}\n\n"
    )
    escalation_instruction = (
        TELEGRAM_ESCALATION_ENABLED_INSTRUCTION
        if ENABLE_TELEGRAM_ESCALATION
        else TELEGRAM_ESCALATION_DISABLED_INSTRUCTION
    )
    prompt = (
        f"{AGENT_PROMPT}\n\n"
        f"{scenario_date}"
        f"БИЗНЕС-ДАННЫЕ:\n\n{business_context}\n\n"
        f"{CLIENT_RESPONSE_INSTRUCTION}\n\n"
        f"{escalation_instruction}"
    )
    if include_source_instruction:
        prompt = f"{prompt}\n\n{SOURCE_INSTRUCTION}"
    if include_action_instruction:
        prompt = f"{prompt}\n\n{ACTION_EXECUTION_INSTRUCTION}"
    return prompt


def parse_tool_permissions(content):
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
        decision = {}
    return {
        "search": decision.get("search") is True,
        "action": decision.get("action") is True,
    }


def decide_tool_permissions():
    config = get_case_config()
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT_PROMPT:\n"
                f"{AGENT_PROMPT}\n\n"
                f"{TOOL_PERMISSION_INSTRUCTION}\n\n"
                "Категория search: поиск в базе знаний.\n"
                "Категория action: "
                f"{config['tool_description']}"
            ),
        },
    ]
    response = complete_messages(messages)
    content = response.choices[0].message.content
    return parse_tool_permissions(content)


def _debug_allowed_tools(permissions):
    if not DEBUG:
        return
    config = get_case_config()
    allowed_tools = []
    if permissions["search"]:
        allowed_tools.append(SEARCH_TOOL_NAME)
    if permissions["action"]:
        allowed_tools.append(config["tool_name"])
    if ENABLE_TELEGRAM_ESCALATION:
        allowed_tools.append(ESCALATION_TOOL_NAME)
    print(
        "DEBUG | Разрешенные инструменты: "
        f"{json.dumps(allowed_tools, ensure_ascii=False)}"
    )


def _debug_tool_call(tool_name, arguments):
    if DEBUG:
        print(
            f"DEBUG | Вызов инструмента {tool_name}: "
            f"{json.dumps(arguments, ensure_ascii=False)}"
        )


def _debug_tool_result(tool_name, result):
    if DEBUG:
        print(
            f"DEBUG | Результат инструмента {tool_name}: "
            f"{json.dumps(result, ensure_ascii=False)}"
        )


def _debug_action_validation(validation):
    if DEBUG:
        print(
            "DEBUG | Семантическая проверка действия: "
            f"{json.dumps(validation, ensure_ascii=False)}"
        )


def _debug_search_scope(search_result):
    if not DEBUG:
        return
    found_sources = [
        result["source"]
        for result in search_result.get("results", [])
    ]
    print(
        "DEBUG | Запрошенные источники: "
        f"{json.dumps(search_result.get('sources', []), ensure_ascii=False)}"
    )
    print(
        "DEBUG | Отсутствующие источники: "
        f"{json.dumps(search_result.get('missing_sources', []), ensure_ascii=False)}"
    )
    print(
        "DEBUG | Найденные источники: "
        f"{json.dumps(found_sources, ensure_ascii=False)}"
    )


def parse_action_validation(content, results):
    content = (content or "").strip()
    try:
        validation = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        json_object = re.search(r"\{.*\}", content, re.DOTALL)
        try:
            validation = json.loads(json_object.group(0)) if json_object else {}
        except (json.JSONDecodeError, TypeError):
            validation = {}

    if not isinstance(validation, dict):
        validation = {}

    decision = validation.get("decision")
    client_response = validation.get("client_response")
    raw_sources = validation.get("used_sources")
    used_sources = _allowed_sources(
        raw_sources if isinstance(raw_sources, list) else [],
        results,
    )

    if decision == "allow":
        return {
            "allowed": True,
            "decision": decision,
            "client_response": "",
            "used_sources": used_sources,
        }

    if (
        decision in {"clarify", "escalate", "reject"}
        and isinstance(client_response, str)
        and client_response.strip()
    ):
        return {
            "allowed": False,
            "decision": decision,
            "client_response": client_response.strip(),
            "used_sources": used_sources,
        }

    return {
        "allowed": False,
        "decision": "invalid",
        "client_response": (
            "Не удалось подтвердить обоснованность действия. "
            "Для продолжения требуется дополнительная проверка специалистом."
        ),
        "used_sources": [],
    }


def validate_action_semantics(
    history,
    user_input,
    business_context,
    results,
    arguments,
):
    config = get_case_config()
    scenario_date = (
        ""
        if DATASET == "own"
        else f"Дата учебного сценария: {SCENARIO_DATE}\n\n"
    )
    found_documents = [
        {
            "source": result.get("source", ""),
            "content": result.get("text", ""),
        }
        for result in results
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT_PROMPT:\n"
                f"{AGENT_PROMPT}\n\n"
                f"{ACTION_VALIDATION_INSTRUCTION}\n\n"
                "ПРЕДПОЛАГАЕМОЕ ДЕЙСТВИЕ:\n"
                f"{config['tool_name']}: {config['tool_description']}\n\n"
                f"{scenario_date}"
                "БИЗНЕС-ДАННЫЕ:\n"
                f"{business_context}\n\n"
                "НАЙДЕННЫЕ ДОКУМЕНТЫ:\n"
                f"{json.dumps(found_documents, ensure_ascii=False)}\n\n"
                "АРГУМЕНТЫ ДЕЙСТВИЯ:\n"
                f"{json.dumps(arguments, ensure_ascii=False)}"
            ),
        },
        *history,
        {"role": "user", "content": user_input},
    ]
    response = complete_messages(messages)
    content = response.choices[0].message.content
    validation = parse_action_validation(content, results)
    _debug_action_validation(validation)
    return validation


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


def validate_escalation_arguments(arguments, results=None, knowledge_used=False):
    if not isinstance(arguments, dict):
        raise ValueError("Аргументы инструмента должны быть объектом.")

    validated = {}
    for field in ("reason", "summary"):
        value = arguments.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Не заполнено обязательное поле: {field}.")
        validated[field] = value.strip()
    protected_markers = (
        "СЛУЖЕБНАЯ ИНСТРУКЦИЯ",
        "AGENT_PROMPT:",
        "SYSTEM PROMPT",
    )
    combined_text = "\n".join(
        [validated["reason"], validated["summary"]]
    ).upper()
    if any(marker in combined_text for marker in protected_markers):
        raise ValueError("Нельзя включать системные инструкции в уведомление.")
    validated["reference_id"] = str(
        arguments.get("reference_id") or ""
    ).strip()

    if knowledge_used:
        raw_sources = arguments.get("used_sources")
        if not isinstance(raw_sources, list):
            raise ValueError("Не заполнено обязательное поле: used_sources.")
        validated["used_sources"] = _allowed_sources(raw_sources, results or [])
        if not validated["used_sources"]:
            raise ValueError("Не указано допустимое основание эскалации.")
    return validated


def build_escalation_message(arguments):
    lines = [
        "Требуется участие специалиста",
        "",
        f"Кейс: {DATASET.capitalize()}",
    ]
    reference_id = arguments.get("reference_id", "")
    if reference_id:
        label = "Заказ" if DATASET == "demo" else "Идентификатор"
        lines.append(f"{label}: {reference_id}")
    lines.extend([
        f"Причина: {arguments['reason']}",
        f"Описание: {arguments['summary']}",
    ])
    return "\n".join(lines)


def execute_escalation(arguments):
    try:
        return send_notification(build_escalation_message(arguments))
    except Exception:
        return {
            "status": "send_error",
            "error": "Не удалось выполнить отправку в Telegram.",
        }


def build_escalation_result_answer(tool_result):
    if tool_result.get("status") == "sent":
        return "Ваше обращение передано специалисту"
    return (
        "Не удалось передать обращение специалисту. "
        "Для продолжения требуется участие специалиста."
    )


def validate_training_lead_match(arguments, history, user_input):
    if DATASET != "training":
        return arguments

    if not ENABLE_LEAD_MATCHING:
        explicit_ids = re.findall(
            get_case_config("training")["id_pattern"],
            get_dialog_text(history, user_input),
        )
        if not explicit_ids:
            raise ValueError(
                "Запись для регистрации не определена. Не запрашивайте у "
                "клиента внутренний идентификатор."
            )
        if _normalize_identity(arguments.get("lead_id")) != _normalize_identity(
            explicit_ids[-1]
        ):
            raise ValueError(
                "Аргументы заявки не соответствуют записи, явно указанной "
                "в диалоге."
            )
        return arguments

    match = match_training_lead(history, user_input)
    if match["status"] != "matched":
        raise ValueError(
            "Лид не сопоставлен однозначно. Уточните имя и компанию или email."
        )

    record = match["record"]
    for field in ("lead_id", "company", "contact"):
        if _normalize_identity(arguments.get(field)) != _normalize_identity(
            record.get(field)
        ):
            raise ValueError(
                "Аргументы заявки не соответствуют однозначно найденному лиду."
            )
        arguments[field] = record[field]
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
    if SIMULATE_TOOL_ERROR:
        return {"status": "tool_error"}

    if DATASET == "demo":
        return create_return_request(
            arguments["order_id"],
            arguments["item"],
            arguments["reason"],
            arguments.get("comment", ""),
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
                "Мы свяжемся с вами для согласования времени."
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
            f"Номер заявки: {tool_result['request_id']}."
        )
    if status == "already_exists":
        return (
            f"По этому случаю уже есть заявка {tool_result['request_id']}. "
            "Новую заявку я не создавал."
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
    normalized = " ".join(text.lower().replace("ё", "е").split())
    normalized = normalized.rstrip(".,!?;:…").rstrip()
    return normalized in phrases


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


def _print_answer(answer, sources=None, response_started=False):
    if response_started:
        print(f"\n{answer}")
    else:
        print(f"\nИИ:\n{answer}")
    _print_sources(sources or [])
    print()


def _handle_pending(user_input, pending, history, config):
    if is_no(user_input):
        answer = "Хорошо, действие отменено."
        _append_exchange(history, user_input, answer)
        print(f"\nИИ:\n{answer}\n")
        return None

    if not is_yes(user_input):
        print(
            "\nИИ:\nПеред выполнением действия нужно подтверждение. "
            "Ответьте да или нет.\n"
        )
        return pending

    arguments = pending["arguments"]
    tool_result = execute_tool(arguments)
    _debug_tool_result(config["tool_name"], tool_result)
    answer = build_tool_result_answer(tool_result)
    _append_exchange(history, user_input, answer)
    _print_tool_call(config["tool_name"], response_started=False)
    _print_answer(
        answer,
        arguments.get("used_sources", []),
        response_started=True,
    )
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


def _print_tool_call(tool_name, response_started):
    if not response_started:
        print("\nИИ:")
    print(f"Инструмент: {tool_name}")
    return True


def _append_tool_result(messages, message, tool_call, result, call_number):
    call_id = getattr(tool_call, "id", None) or f"call_{call_number}"
    _debug_tool_result(tool_call.function.name, result)
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
    permissions = decide_tool_permissions()
    _debug_allowed_tools(permissions)
    business_context = find_business_context(history, user_input)
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                business_context,
                include_source_instruction=permissions["search"],
                include_action_instruction=permissions["action"],
            ),
        },
        *history,
        {"role": "user", "content": user_input},
    ]
    knowledge_available = bool(documents)
    search_attempted = False
    response_started = False
    results = []

    for call_number in range(1, MAX_TOOL_CALLS + 1):
        response = complete_messages(
            messages,
            tools=build_tools(
                results,
                knowledge_available,
                search_allowed=permissions["search"],
                action_allowed=permissions["action"],
            ),
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
                "response_started": response_started,
            }

        try:
            tool_name, arguments = _parse_function_call(tool_call)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            tool_name = getattr(tool_call.function, "name", "unknown")
            response_started = _print_tool_call(
                tool_name,
                response_started,
            )
            _debug_tool_call(
                tool_name,
                getattr(tool_call.function, "arguments", None),
            )
            result = {"status": "invalid_arguments", "error": str(error)}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        response_started = _print_tool_call(tool_name, response_started)
        _debug_tool_call(tool_name, arguments)

        if tool_name == SEARCH_TOOL_NAME:
            if not permissions["search"]:
                result = {
                    "status": "tool_not_allowed",
                    "tool": tool_name,
                }
                _append_tool_result(
                    messages, message, tool_call, result, call_number,
                )
                continue

            search_attempted = True
            search_result = search_knowledge_base(
                arguments.get("query", ""),
                documents,
                frequencies,
                sources=arguments.get("sources"),
            )
            _debug_search_scope(search_result)
            results = _merge_results(results, search_result["results"])
            _append_tool_result(
                messages, message, tool_call, search_result, call_number,
            )
            continue

        if tool_name == ESCALATION_TOOL_NAME:
            if not ENABLE_TELEGRAM_ESCALATION:
                result = {
                    "status": "tool_not_allowed",
                    "tool": tool_name,
                }
                _append_tool_result(
                    messages, message, tool_call, result, call_number,
                )
                continue
            if search_attempted and not results:
                result = {
                    "status": "insufficient_knowledge",
                    "message": (
                        "База знаний не подтвердила основание эскалации. "
                        "Уведомление не отправлено."
                    ),
                }
                _append_tool_result(
                    messages, message, tool_call, result, call_number,
                )
                continue
            try:
                validated = validate_escalation_arguments(
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

            result = execute_escalation(validated)
            _debug_tool_result(tool_name, result)
            return {
                "type": "answer",
                "answer": build_escalation_result_answer(result),
                "used_sources": validated.get("used_sources", []),
                "response_started": response_started,
            }

        if tool_name != get_case_config()["tool_name"]:
            result = {"status": "unknown_tool", "tool": tool_name}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        if not permissions["action"]:
            result = {
                "status": "tool_not_allowed",
                "tool": tool_name,
            }
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
            validated = validate_training_lead_match(
                validated,
                history,
                user_input,
            )
        except ValueError as error:
            result = {"status": "invalid_arguments", "error": str(error)}
            _append_tool_result(
                messages, message, tool_call, result, call_number,
            )
            continue

        semantic_validation = validate_action_semantics(
            history,
            user_input,
            business_context,
            results,
            validated,
        )
        if not semantic_validation["allowed"]:
            if (
                ENABLE_TELEGRAM_ESCALATION
                and semantic_validation["decision"] == "escalate"
            ):
                result = {
                    "status": "escalation_required",
                    "message": semantic_validation["client_response"],
                    "used_sources": semantic_validation["used_sources"],
                }
                _append_tool_result(
                    messages, message, tool_call, result, call_number,
                )
                continue
            return {
                "type": "answer",
                "answer": semantic_validation["client_response"],
                "used_sources": semantic_validation["used_sources"],
                "response_started": response_started,
            }

        _debug_tool_result(tool_name, {
            "status": "awaiting_confirmation",
            "arguments": validated,
        })
        return {
            "type": "action",
            "arguments": validated,
            "response_started": response_started,
        }

    return {
        "type": "answer",
        "answer": (
            "Не удалось завершить обработку запроса: превышен лимит "
            "последовательных вызовов инструментов."
        ),
        "used_sources": [],
        "response_started": response_started,
    }


def main():
    clear_terminal()
    config = get_case_config()
    print("Подключение к Google Sheets...", end="", flush=True)
    try:
        prepare_case_worksheet(config)
    except Exception as error:
        print()
        print(
            "Не удалось подготовить Google Sheets: "
            f"{error}\nЗапуск Agent остановлен."
        )
        return
    print("\r\033[2K", end="", flush=True)
    documents, frequencies = _load_optional_knowledge_base()

    history = []
    pending = None
    print_startup_header("Agent", DATASET)
    if documents:
        print(f"База знаний: {len(documents)} файлов.")
    else:
        print("База знаний: не подключена.")
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
            if pending:
                user_input = input("Вы: ").strip()
            else:
                user_input = session.prompt("Вы:\n")
        except KeyboardInterrupt:
            print()
            break

        if user_input.strip().lower() == "/exit":
            break
        if not user_input.strip():
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
            _print_answer(
                answer,
                used_sources,
                response_started=turn.get("response_started", False),
            )
            continue

        arguments = turn["arguments"]
        confirmation = make_confirmation(arguments)
        _append_exchange(history, user_input, confirmation)
        pending = {"arguments": arguments}
        _print_answer(
            confirmation,
            response_started=turn.get("response_started", False),
        )


if __name__ == "__main__":
    main()
