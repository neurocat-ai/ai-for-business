import os
from pathlib import Path
import re

import gspread
from dotenv import load_dotenv
from gspread.exceptions import WorksheetNotFound


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "").strip()

WORKSHEET_NAMES = {
    "demo": "Demo",
    "training": "Training",
    "own": "Own",
}

SERVICE_ACCOUNT_FILE = (
    ROOT
    / "infrastructure"
    / "service_account.json"
)


def get_worksheet(dataset):
    try:
        worksheet_name = WORKSHEET_NAMES[dataset]
    except KeyError as error:
        raise ValueError(
            f"Неизвестный кейс Google Sheets: {dataset}"
        ) from error

    if not SPREADSHEET_ID:
        raise RuntimeError(
            "Не задана обязательная настройка "
            "Google Sheets: SPREADSHEET_ID. "
            "Заполните .env в корне проекта "
            "по образцу .env.example."
        )

    client = gspread.service_account(
        filename=str(SERVICE_ACCOUNT_FILE)
    )

    spreadsheet = client.open_by_key(
        SPREADSHEET_ID
    )

    try:
        return spreadsheet.worksheet(
            worksheet_name
        )
    except WorksheetNotFound as error:
        raise RuntimeError(
            "В Google-таблице не найдена "
            f"вкладка '{worksheet_name}' "
            f"для кейса '{dataset}'."
        ) from error


def read_rows(worksheet):
    return worksheet.get_all_records()


def write_rows(
    worksheet,
    headers,
    rows,
):
    worksheet.update(
        [headers, *rows],
        "A1",
    )


def append_row(
    worksheet,
    row,
):
    worksheet.append_row(row)


def initialize_worksheet(
    worksheet,
    headers,
    initial_rows,
    optional_fields=(),
    request_id_prefix=None,
):
    current_values = worksheet.get_all_values()

    if not any(
        str(cell).strip()
        for row in current_values
        for cell in row
    ):
        write_rows(
            worksheet,
            headers,
            initial_rows,
        )
        return

    invalid_data_message = (
        "Вкладка Google Sheets заполнена некорректно. "
        "Проверьте данные или очистите вкладку и перезапустите агент."
    )
    current_headers = current_values[0] if current_values else []
    if current_headers != headers:
        raise RuntimeError(invalid_data_message)

    data_rows = [
        row
        for row in current_values[1:]
        if any(str(cell).strip() for cell in row)
    ]
    if not data_rows:
        if initial_rows:
            worksheet.update(initial_rows, "A2")
        return

    optional_fields = set(optional_fields)
    required_indexes = [
        index
        for index, header in enumerate(headers)
        if header not in optional_fields
    ]
    request_id_index = headers.index("request_id")
    request_ids = set()

    for raw_row in data_rows:
        if (
            len(raw_row) > len(headers)
            and any(str(cell).strip() for cell in raw_row[len(headers):])
        ):
            raise RuntimeError(invalid_data_message)

        row = [*raw_row[:len(headers)]]
        row.extend([""] * (len(headers) - len(row)))
        if any(not str(row[index]).strip() for index in required_indexes):
            raise RuntimeError(invalid_data_message)

        request_id = str(row[request_id_index]).strip()
        if (
            request_id in request_ids
            or request_id_prefix
            and not re.fullmatch(
                rf"{re.escape(request_id_prefix)}-\d+",
                request_id,
            )
        ):
            raise RuntimeError(invalid_data_message)
        request_ids.add(request_id)

    initial_request_ids = {
        str(row[request_id_index]).strip()
        for row in initial_rows
    }
    if not initial_request_ids.issubset(request_ids):
        raise RuntimeError(
            invalid_data_message
        )


def clear_worksheet(worksheet):
    worksheet.clear()
