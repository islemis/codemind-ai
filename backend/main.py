from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.local_project_service import load_local_project
from backend.analyzers.tree_sitter_analyzer import analyze_repository

app = FastAPI(title="CodeMind AI")


class ProjectRequest(BaseModel):
    project_path: str


@app.get("/")
def home():
    return {"message": "CodeMind AI is running"}


@app.post("/projects/import")
def import_local_project(request: ProjectRequest):
    try:
        project = load_local_project(request.project_path)

        components = analyze_repository(
            project["project_path"]
        )

        return {
            "message": "Project analyzed successfully",
            "project_name": project["project_name"],
            "total_components": len(components),
            "components": components[:20],
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))