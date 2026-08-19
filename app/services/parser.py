from pathlib import Path

from pypdf import PdfReader

# openpyxl이 읽는 형식만. 구형 .xls는 지원하지 않는다.
_SPREADSHEET_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def is_spreadsheet(path: Path, content_type: str) -> bool:
    """표 문서인지 판별. 표는 문장 패킹 대신 행 단위로 청킹해야 한다(services/table.py)."""
    return content_type in _SPREADSHEET_TYPES or path.suffix.lower() in {".xlsx", ".xlsm"}


def extract_text(path: Path, content_type: str) -> str:
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # 기본은 텍스트로 취급 (txt, md 등)
    return path.read_text(encoding="utf-8", errors="ignore")
