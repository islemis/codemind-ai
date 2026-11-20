
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import models

from backend.local_project_service import load_local_project
from backend.analyzers.chunking_analyzer import analyze_project

from backend.services.qdrant_service import (
    client,
    COLLECTION_NAME,
    index_components,
    search_components,
)

from backend.services.chat_service import chat_with_code


app = FastAPI(title="CodeMind AI")


# Modèle pour sélectionner un projet local
class ProjectRequest(BaseModel):
    project_path: str


# Modèle pour rechercher dans Qdrant
class SearchRequest(ProjectRequest):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


# Modèle pour poser une question au chatbot
class ChatRequest(ProjectRequest):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


# Vérifier que l'API fonctionne
@app.get("/")
def health():
    return {
        "message": "CodeMind AI is running"
    }


# Analyser et indexer un projet
@app.post("/projects/index")
def index_project(request: ProjectRequest):

    try:
        # 1. Vérifier que le projet local est autorisé
        project = load_local_project(
            request.project_path
        )

        project_path = str(
            Path(project["project_path"]).resolve()
        )

        # 2. Analyser le projet avec LangChain
        components = analyze_project(project_path)

        if not components:
            raise ValueError(
                "No text chunks found in this project."
            )

        # 3. Supprimer les anciens chunks
        # uniquement pour ce projet
        if client.collection_exists(COLLECTION_NAME):

            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="project_path",
                                match=models.MatchValue(
                                    value=project_path
                                )
                            )
                        ]
                    )
                ),
                wait=True
            )

        # 4. Générer les embeddings
        # et enregistrer les chunks dans Qdrant
        result = index_components(
            project_path,
            components
        )

        return {
            "message": "Project indexed successfully",
            "project_name": project["project_name"],
            "total_components": len(components),
            "index": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error


# Tester directement la recherche dans Qdrant
@app.post("/projects/search")
def search_project(request: SearchRequest):

    try:
        project = load_local_project(
            request.project_path
        )

        results = search_components(
            project["project_path"],
            request.query,
            top_k=request.top_k
        )

        return {
            "results": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error


# Poser une question à CodeMind AI
@app.post("/chat")
def chat_project(request: ChatRequest):

    try:
        project = load_local_project(
            request.project_path
        )

        return chat_with_code(
            project["project_path"],
            request.question,
            top_k=request.top_k
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
