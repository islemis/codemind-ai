
import json
from pathlib import Path

from langchain_core.tools import tool

from backend.analyzers.file_scanner import scan_project_files
from backend.services.qdrant_service import search_components





def build_project_tools(project_path: str, top_k: int = 5):
    root = Path(project_path).resolve()

    # Only allow files selected by our scanner
    files = {
        path.relative_to(root).as_posix(): path
        for path in scan_project_files(str(root))
    }

    @tool
    def list_project_files(name_filter: str = "") -> str:
        """List available project files. Optionally filter by filename or extension."""
        matches = [
            name for name in sorted(files)
            if name_filter.lower() in name.lower()
        ]

        return json.dumps({
            "total": len(matches),
            "files": matches[:150]
        })

    @tool
    def read_project_file(file_path: str, start_line: int = 1) -> str:
        """Read an allowed project file by relative path. Use start_line to continue reading long files."""
        path = files.get(file_path.replace("\\", "/"))

        if not path or not path.resolve().is_relative_to(root):
            return "File not found or not allowed."

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "Could not read file."

        if "\x00" in text:
            return "File blocked: binary  content."

        lines = text.splitlines()

        if start_line < 1 or start_line > max(1, len(lines)):
            return "Invalid start line."

        selected = lines[start_line - 1:start_line + 199]

        content = "\n".join(
            f"{start_line + i}: {line}"
            for i, line in enumerate(selected)
        )

        return json.dumps({
            "file": file_path,
            "start_line": start_line,
            "end_line": start_line + len(selected) - 1,
            "total_lines": len(lines),
            "content": content[:16000],
            "has_more": start_line + len(selected) - 1 < len(lines)
        })

    @tool
    def search_code(query: str) -> str:
        """Search indexed code, documentation and configuration in Qdrant."""
        try:
            results = search_components(
                str(root), query, top_k=top_k
            )
        except ValueError as error:
            return str(error)

        safe_results = []

        for item in results:
            if item["file"] not in files:
                continue

            code = item.get("code", "")

           

            safe_results.append({
                "file": item["file"],
                "name": item.get("name"),
                "type": item.get("type"),
                "start_line": item["start_line"],
                "end_line": item["end_line"],
                "score": item["score"],
                "code": code[:1500]
            })

        return json.dumps({"results": safe_results})

    return [
        list_project_files,
        read_project_file,
        search_code
    ]
