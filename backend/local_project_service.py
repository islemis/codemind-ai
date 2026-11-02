from pathlib import Path

# Directory containing the local projects
PROJECTS_DIR = (Path.home() / "Desktop" / "projects").resolve()
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def load_local_project(project_path: str):
    path = Path(project_path).resolve()

    # Validate the project directory
    if not path.is_dir():
        raise ValueError("Project directory not found.")

    # Restrict access to the allowed projects directory
    if path == PROJECTS_DIR or not path.is_relative_to(PROJECTS_DIR):
        raise ValueError("Project must be inside the projects directory.")

    return {
        "project_name": path.name,
        "project_path": str(path)
    }