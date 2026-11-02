
from pathlib import Path
from tree_sitter_language_pack import get_parser

from backend.analyzers.language_config import LANGUAGES, NODE_TYPES


IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build"
}


def analyze_repository(project_path: str):
    root = Path(project_path).resolve()
    components = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)

        if any(part in IGNORED_DIRS for part in relative_path.parts):
            continue

        if not file_path.resolve().is_relative_to(root):
            continue

        language = LANGUAGES.get(file_path.suffix.lower())

        if not language or file_path.stat().st_size > 200_000:
            continue

        try:
            source = file_path.read_bytes()
            parser = get_parser(language)
            tree = parser.parse(source)

            # Traverse the syntax tree
            stack = [tree.root_node]

            while stack:
                node = stack.pop()
                component_type = NODE_TYPES[language].get(node.type)

                if component_type:
                    name_node = node.child_by_field_name("name")

                    name = (
                        source[name_node.start_byte:name_node.end_byte]
                        .decode("utf-8", errors="replace")
                        if name_node else None
                    )

                    code = source[
                        node.start_byte:node.end_byte
                    ].decode("utf-8", errors="replace")

                    components.append({
                        "file": relative_path.as_posix(),
                        "language": language,
                        "name": name,
                        "type": component_type,
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "code": code,
                    })

                stack.extend(reversed(node.named_children))

        except (OSError, ValueError, RuntimeError) as error:
            print(f"Could not analyze {relative_path}: {error}")

    return components
