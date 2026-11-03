
from pathlib import Path
from tree_sitter_language_pack import get_parser

from backend.analyzers.language_config import LANGUAGES, NODE_TYPES
from backend.analyzers.file_scanner import scan_project_files


IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build"
}

# Import nodes for Python, Java, JavaScript and TypeScript
IMPORT_NODE_TYPES = {
    "import_statement",
    "import_from_statement",
    "import_declaration"
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

    for file_path in scan_project_files(root):
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

            stack = [tree.root_node]

            # Collect imports separately for this file
            import_nodes = []

            while stack:
                node = stack.pop()

                # Group imports instead of creating one chunk each
                if node.type in IMPORT_NODE_TYPES:
                    import_nodes.append(node)
                    continue

                component_type = NODE_TYPES[language].get(node.type)

                if component_type:
                    name = get_node_name(node, source)
                    parent_class = get_parent_class(node, source)

                    if (
                        language == "python"
                        and component_type == "function"
                        and parent_class
                    ):
                        component_type = "method"

                    start_byte = node.start_byte
                    end_byte = node.end_byte
                    end_line = node.end_point.row + 1

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

            # Create ONE import chunk per file
            if import_nodes:
                import_nodes.sort(key=lambda node: node.start_byte)

                imports_code = "\n".join(
                    source[
                        node.start_byte:node.end_byte
                    ].decode("utf-8", errors="replace")
                    for node in import_nodes
                )

                components.append({
                    "file": relative_path.as_posix(),
                    "language": language,
                    "name": "imports",
                    "type": "import",
                    "parent_class": None,
                    "start_line": import_nodes[0].start_point.row + 1,
                    "end_line": import_nodes[-1].end_point.row + 1,
                    "code": imports_code,
                })

        except (OSError, ValueError, RuntimeError) as error:
            print(f"Could not analyze {relative_path}: {error}")

    return components
