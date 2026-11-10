from pathlib import Path

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from backend.analyzers.file_scanner import scan_project_files


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Quelques extensions dont le nom est different du nom LangChain.
# Les autres (.java, .go, .php, .swift...) fonctionnent directement.
ALIASES = {
    ".py": "python",
    ".jsx": "js",
    ".tsx": "ts",
    ".rs": "rust",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".md": "markdown",
    ".htm": "html",
    ".tex": "latex",
    ".hs": "haskell",
    ".ps1": "powershell",
}

# Liste officielle des langages supportes par ta version installee
SUPPORTED_LANGUAGES = {lang.value: lang for lang in Language}


def get_splitter(file_path: Path):
    """Specialise pour un langage connu, generique pour le reste."""
    extension = file_path.suffix.lower()
    language_name = ALIASES.get(extension, extension.lstrip("."))
    language = SUPPORTED_LANGUAGES.get(language_name)

    settings = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "add_start_index": True,
    }

    if language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=language, **settings
        )

    # JSON, YAML, XML, TXT et extensions inconnues
    return RecursiveCharacterTextSplitter(**settings)


def analyze_project(project_path: str):
    """Analyser tous les textes et conserver les lignes originales."""
    root = Path(project_path).resolve()
    components = []

    for file_path in scan_project_files(str(root)):
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if not content.strip():
            continue

        relative_path = file_path.relative_to(root).as_posix()
        extension = file_path.suffix.lower()
        language_name = ALIASES.get(extension, extension.lstrip(".")) or "text"

        # LangChain decoupe le texte selon sa structure, si connue
        chunks = get_splitter(file_path).create_documents([content])

        for i, chunk in enumerate(chunks, start=1):
            code = chunk.page_content
            start = chunk.metadata.get("start_index")

            # Eviter les fausses citations si la position est introuvable
            if (
                not code
                or not isinstance(start, int)
                or start < 0
                or content[start:start + len(code)] != code
            ):
                print(f"Chunk sans position fiable ignore : {relative_path} #{i}")
                continue

            # Calculer les numeros des lignes dans le vrai fichier
            start_line = content.count("\n", 0, start) + 1
            end_line = content.count("\n", 0, start + len(code) - 1) + 1

            # Meme structure que l'ancien index, donc compatible Qdrant
            components.append({
                "file": relative_path,
                "language": language_name,
                "name": f"{file_path.name}#chunk_{i}",
"type": "code" if language_name in SUPPORTED_LANGUAGES else "document",
                "parent_class": None,
                "start_line": start_line,
                "end_line": end_line,
                "code": code,
            })

    return components
