from pathlib import Path

import gspread
from gspread.exceptions import WorksheetNotFound


ROOT = Path(__file__).resolve().parents[1]

SPREADSHEET_ID = (
    "ВСТАВЬТЕ_ID_ТАБЛИЦЫ"
)

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
):
    current_headers = (
        worksheet.row_values(1)
    )

    if not current_headers:
        write_rows(
            worksheet,
            headers,
            initial_rows,
        )
        return

    if current_headers[:len(headers)] != headers:
        raise RuntimeError(
            "Структура вкладки "
            "не соответствует ожидаемой."
        )


def clear_worksheet(worksheet):
    worksheet.clear()
