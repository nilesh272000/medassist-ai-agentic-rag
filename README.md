# MedAssist AI — Agentic Medical Knowledge Assistant

Portfolio-focused medical knowledge assistant demonstrating Agentic RAG, LangGraph, FastAPI, OpenAI, ChromaDB, evidence grading, query rewriting, safe fallback, multimodal endpoints, speech I/O and a browser UI.

> Educational engineering project only. It is not a diagnostic or clinical system.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# add OPENAI_API_KEY
python ingest.py
python run.py
```
UI: http://127.0.0.1:8000  
Swagger: http://127.0.0.1:8000/docs  
Health: http://127.0.0.1:8000/api/health

## Agent flow
User → Intent Router → Vector Retrieval → Evidence Grader → (GOOD → Grounded Answer) / (WEAK → Query Rewrite → Retry → Safe Fallback)

## Features
- Agentic RAG with conditional LangGraph workflow
- ChromaDB vector retrieval
- Evidence validation before generation
- Query rewrite/retry
- Grounded source citations
- Safe insufficient-evidence response
- Optional Tavily configuration
- Image observation, speech-to-text and text-to-speech endpoints
- FastAPI + Swagger
- Browser UI
- Tests and Dockerfile

## Safety
Do not use this application to diagnose, prescribe, or make emergency decisions. Do not send identifiable patient information to third-party APIs without appropriate authorization. Use current authoritative clinical references for real medical decisions.
