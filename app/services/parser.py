from pathlib import Path

from pypdf import PdfReader


def extract_text(path: Path, content_type: str) -> str:
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    # 기본은 텍스트로 취급 (txt, md 등)
    return path.read_text(encoding="utf-8", errors="ignore")
