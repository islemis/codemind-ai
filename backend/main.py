
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.services.chat_service import chat_with_code
from backend.analyzers.document_analyzer import analyze_documents
from backend.local_project_service import load_local_project
from backend.analyzers.tree_sitter_analyzer import analyze_repository
from backend.services.qdrant_service import (
    index_components,
    search_components
)


app = FastAPI(title="CodeMind AI")


class ProjectRequest(BaseModel):
    project_path: str


class SearchRequest(BaseModel):
    project_path: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/")
def home():
    return {"message": "CodeMind AI is running"}



@app.post("/projects/index")
def index_local_project(request: ProjectRequest):
    try:
        project = load_local_project(request.project_path)
        components = analyze_repository(project["project_path"])
        components += analyze_documents(project["project_path"])

        result = index_components(
            project["project_path"],
            components
        )

        return {
            "message": "Project indexed successfully",
            "project_name": project["project_name"],
            **result
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/projects/search")
def search_local_project(request: SearchRequest):
    try:
        project = load_local_project(request.project_path)

        results = search_components(
            project["project_path"],
            request.query,
            request.top_k
        )

        return {
            "total_results": len(results),
            "results": results
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))



class ChatRequest(BaseModel):
    project_path: str
    question: str = Field(min_length=1)


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        project = load_local_project(request.project_path)

        return chat_with_code(
            project_path=project["project_path"],
            question=request.question,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))