import ast
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".venv", "venv",
    "__pycache__", "node_modules", ".pytest_cache"
}


def analyze_repository(project_path: str):
    root = Path(project_path).resolve()
    results = []

    for file_path in root.rglob("*.py"):
        relative_path = file_path.relative_to(root)

        if any(
            part in IGNORED_DIRS or part.startswith(".")
            for part in relative_path.parts
        ):
            continue

        # Ne pas suivre les liens symboliques vers d'autres dossiers
        if not file_path.resolve().is_relative_to(root):
            continue

        if not file_path.is_file() or file_path.stat().st_size > 200_000:
            continue

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (SyntaxError, UnicodeError, OSError):
            continue

        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                results.append({
                    "file": relative_path.as_posix(),
                    "name": node.name,
                    "type": (
                        "class" if isinstance(node, ast.ClassDef)
                        else "function"
                    ),
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "code": ast.get_source_segment(source, node),
                })

    return results