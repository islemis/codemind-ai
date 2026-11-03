
import os
from pathlib import Path
import pathspec

from backend.analyzers.language_config import LANGUAGES


DOCUMENT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".rst",
    ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".properties",
    ".xml", ".sh", ".ps1", ".bat", ".sql",
    ".tf", ".hcl"
}

SPECIAL_FILES = {
    "dockerfile", "makefile", "jenkinsfile",
    "procfile", ".gitignore", ".dockerignore",
    ".editorconfig", ".gitmodules",
    "readme", "license", "notice",
    "cmakelists.txt", "gemfile", "rakefile"
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build", ".next",
    ".cache", "coverage", "target",
    ".pytest_cache", ".mypy_cache",
    ".gradle", "vendor"
}

IGNORED_FILES = {
    "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "poetry.lock", "uv.lock",
    "pipfile.lock", "cargo.lock",
    "credentials.json", "secrets.yaml",
    "secrets.yml", "id_rsa", "id_ed25519"
}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

MAX_FILE_BYTES = 200_000


def is_document_file(path: Path) -> bool:
    name = path.name.lower()

    return (
        name in SPECIAL_FILES
        or name.startswith("dockerfile")
        or path.suffix.lower() in DOCUMENT_EXTENSIONS
    )


def scan_project_files(project_path: str):
    root = Path(project_path).resolve()

    if not root.is_dir():
        raise ValueError("Project directory not found.")

    # Load root .gitignore
    ignore_file = root / ".gitignore"

    ignore_lines = (
        ignore_file.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()
        if ignore_file.is_file() and not ignore_file.is_symlink()
        else []
    )

    ignore_spec = pathspec.PathSpec.from_lines(
        "gitwildmatch",
        ignore_lines
    )

    for directory, dirs, files in os.walk(
        root, followlinks=False
    ):
        current = Path(directory)

        # Skip ignored directories
        dirs[:] = sorted(
            name for name in dirs
            if name not in IGNORED_DIRS
            and not (current / name).is_symlink()
            and not ignore_spec.match_file(
                (current / name).relative_to(root).as_posix() + "/"
            )
        )

        for name in sorted(files):
            file_path = current / name
            lower_name = name.lower()

            # Skip sensitive files
            if (
                file_path.is_symlink()
                or lower_name.startswith(".env")
                or lower_name in IGNORED_FILES
                or file_path.suffix.lower() in SECRET_SUFFIXES
            ):
                continue

            # Keep supported source and text files
            if not (
                file_path.suffix.lower() in LANGUAGES
                or is_document_file(file_path)
            ):
                continue

            relative = file_path.relative_to(root).as_posix()

            if ignore_spec.match_file(relative):
                continue

            try:
                if file_path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue

            yield file_path
