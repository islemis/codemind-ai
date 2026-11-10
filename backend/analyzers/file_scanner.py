import os
from pathlib import Path

import pathspec


# Dossiers inutiles pour comprendre le projet
IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", ".next", ".cache", "coverage", "target",
    ".pytest_cache", ".mypy_cache", ".gradle", "vendor"
}

# Fichiers generes automatiquement (lock files)
IGNORED_FILES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "poetry.lock", "uv.lock", "pipfile.lock", "cargo.lock"
}

# Noms de fichiers susceptibles de contenir des identifiants
SENSITIVE_FILES = {
    "credentials.json", "credentials.yaml", "credentials.yml",
    "secrets.json", "secrets.yaml", "secrets.yml",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials",
    ".envrc", "application.properties", "application.yml",
    "application.yaml", "bootstrap.properties", "bootstrap.yml",
    "bootstrap.yaml"
}

SENSITIVE_EXTENSIONS = {
    ".pem", ".key", ".p12", ".pfx", ".jks",
    ".keystore", ".tfstate"
}

# On ignore les fichiers trop gros pour cette V1 (200 Ko)
MAX_FILE_BYTES = 200_000


def is_sensitive_file(path: Path) -> bool:
    """Exclure des fichiers sensibles par leur NOM (sans lire les secrets)."""
    name = path.name.lower()

    if name == ".env" or name.startswith(".env."):
        return True

    if name in SENSITIVE_FILES or path.suffix.lower() in SENSITIVE_EXTENSIONS:
        return True

    if name.endswith(".tfstate.backup"):
        return True

    # Configurations Spring Boot selon l'environnement
    if name.startswith(("application-", "bootstrap-")):
        if path.suffix.lower() in {".properties", ".yml", ".yaml"}:
            return True

    return False


def is_text_file(path: Path) -> bool:
    """Accepter les textes UTF-8, sans liste obligatoire d'extensions."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return False

        data = path.read_bytes()

        if not data or b"\x00" in data:
            return False

        data.decode("utf-8")
        return True

    except (OSError, UnicodeDecodeError):
        return False


def scan_project_files(project_path: str):
    """Parcourir automatiquement tous les fichiers texte autorises."""
    root = Path(project_path).resolve()

    if not root.is_dir():
        raise ValueError("Project directory not found.")

    # Respecter le .gitignore a la racine du projet
    ignore_file = root / ".gitignore"
    ignore_lines = (
        ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if ignore_file.is_file() and not ignore_file.is_symlink()
        else []
    )
    ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_lines)

    for directory, dirs, filenames in os.walk(root, followlinks=False):
        current = Path(directory)

        # Ne pas entrer dans les dossiers exclus ou les liens symboliques
        dirs[:] = sorted(
            name for name in dirs
            if name.lower() not in IGNORED_DIRS
            and not (current / name).is_symlink()
            and not ignore_spec.match_file(
                (current / name).relative_to(root).as_posix() + "/"
            )
        )

        for name in sorted(filenames):
            file_path = current / name

            if (
                file_path.is_symlink()
                or name.lower() in IGNORED_FILES
                or is_sensitive_file(file_path)
            ):
                continue

            relative_path = file_path.relative_to(root).as_posix()

            if ignore_spec.match_file(relative_path):
                continue

            # Tous les fichiers texte UTF-8 sont acceptes (.py, .go, .txt, .yaml...)
            if is_text_file(file_path):
                yield file_path
