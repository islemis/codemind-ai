
from pathlib import Path

from backend.analyzers.language_config import LANGUAGES
from backend.analyzers.file_scanner import (
    scan_project_files,
    is_document_file
)


MAX_CHARS = 1800
OVERLAP_LINES = 2


def analyze_documents(project_path: str):
    root = Path(project_path).resolve()
    documents = []

    for file_path in scan_project_files(root):

        # Source code is already handled by Tree-sitter
        if (
            file_path.suffix.lower() in LANGUAGES
            or not is_document_file(file_path)
        ):
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        except OSError:
            continue

        # Skip binary content
        if "\x00" in content:
            continue

        lines = content.splitlines()
        relative = file_path.relative_to(root)

        start = 0
        chunk_number = 1

        while start < len(lines):
            end = start
            size = 0

            # Build a chunk using complete lines
            while end < len(lines) and (
                end == start
                or size + len(lines[end]) + 1 <= MAX_CHARS
            ):
                size += len(lines[end]) + 1
                end += 1

            chunk = "\n".join(lines[start:end]).strip()

            if chunk:
                documents.append({
                    "file": relative.as_posix(),
                    "language": (
                        file_path.suffix.lower().lstrip(".")
                        or "text"
                    ),
                    "name": f"{relative.name}#chunk_{chunk_number}",
                    "type": "document",
                    "parent_class": None,
                    "start_line": start + 1,
                    "end_line": end,
                    "code": chunk
                })

                chunk_number += 1

            if end == len(lines):
                break

            # Keep a small overlap between chunks
            start = max(start + 1, end - OVERLAP_LINES)

    return documents
