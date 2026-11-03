
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

# Generated files that are not useful for code understanding
IGNORED_FILES = {
    "package-lock.json", "pnpm-lock.yaml",
    "yarn.lock", "poetry.lock", "uv.lock",
    "pipfile.lock", "cargo.lock"
}

# Known sensitive filenames
SENSITIVE_FILES = {
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".git-credentials",
    ".envrc",
    "application.properties",
    "application.yml",
    "application.yaml",
    "bootstrap.properties",
    "bootstrap.yml",
    "bootstrap.yaml"
}

# Private keys, certificates and keystores
SENSITIVE_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx",
    ".jks", ".keystore", ".tfstate"
}

MAX_FILE_BYTES = 200_000


def is_sensitive_file(path: Path) -> bool:
    name = path.name.lower()

    # Environment files
    if name == ".env" or name.startswith(".env."):
        return True

    # Explicit sensitive filenames
    if name in SENSITIVE_FILES:
        return True

    # Sensitive file extensions
    if path.suffix.lower() in SENSITIVE_EXTENSIONS:
        return True

    # Terraform state backups
    if name.endswith(".tfstate.backup"):
        return True

    # Spring Boot profile-specific configuration
    # Examples: application-dev.properties, application-prod.yml
    if name.startswith(("application-", "bootstrap-")):
        if path.suffix.lower() in {".properties", ".yml", ".yaml"}:
            return True

    return False


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
            if name.lower() not in IGNORED_DIRS
            and not (current / name).is_symlink()
            and not ignore_spec.match_file(
                (current / name).relative_to(root).as_posix() + "/"
            )
        )

        for name in sorted(files):
            file_path = current / name
            lower_name = name.lower()

            # Skip links, generated files and sensitive files
            if (
                file_path.is_symlink()
                or lower_name in IGNORED_FILES
                or is_sensitive_file(file_path)
            ):
                continue

            # Keep supported source and document files
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
