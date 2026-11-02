
from pathlib import Path
from tree_sitter_language_pack import get_parser

from backend.analyzers.language_config import LANGUAGES, NODE_TYPES


IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build"
}


def get_node_name(node, source):
    name_node = node.child_by_field_name("name")

    if name_node:
        return source[
            name_node.start_byte:name_node.end_byte
        ].decode("utf-8", errors="replace")

    return None


def get_parent_class(node, source):
    parent = node.parent

    while parent:
        if parent.type in {"class_definition", "class_declaration"}:
            return get_node_name(parent, source)

        parent = parent.parent

    return None


def analyze_repository(project_path: str):
    root = Path(project_path).resolve()
    components = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)

        # Skip generated and dependency directories
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

            stack = [tree.root_node]

            while stack:
                node = stack.pop()
                component_type = NODE_TYPES[language].get(node.type)

                if component_type:
                    name = get_node_name(node, source)
                    parent_class = get_parent_class(node, source)

                    # Identify Python methods inside classes
                    if (
                        language == "python"
                        and component_type == "function"
                        and parent_class
                    ):
                        component_type = "method"

                    # Use the full syntax node range
                    start_byte = node.start_byte
                    end_byte = node.end_byte
                    end_line = node.end_point.row + 1

                    # Include the complete Python class body
                    if (
                        language == "python"
                        and node.type == "class_definition"
                    ):
                        body = node.child_by_field_name("body")

                        if body and body.end_byte > end_byte:
                            end_byte = body.end_byte
                            end_line = body.end_point.row + 1

                    code = source[
                        start_byte:end_byte
                    ].decode("utf-8", errors="replace")

                    components.append({
                        "file": relative_path.as_posix(),
                        "language": language,
                        "name": name,
                        "type": component_type,
                        "parent_class": parent_class,
                        "start_line": node.start_point.row + 1,
                        "end_line": end_line,
                        "code": code,
                    })

                stack.extend(reversed(node.named_children))

        except (OSError, ValueError, RuntimeError) as error:
            print(f"Could not analyze {relative_path}: {error}")

    return components
