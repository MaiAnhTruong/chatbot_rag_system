#/home/truong/IDT/RAG/chatbot_rag_system/src/app/dependencies.py
from __future__ import annotations
from fastapi import Request
from rag.orchestrator import Orchestrator

def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator
