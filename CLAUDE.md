# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot that enables users to ask natural language questions about course materials. The system uses semantic search to find relevant content and Claude AI to generate context-aware responses with source attribution.

## Key Technologies

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| AI Model | Anthropic Claude (claude-sonnet-4-20250514) |
| Vector Database | ChromaDB |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Package Manager | uv |
| Python Version | 3.13+ |
| Frontend | Vanilla HTML/CSS/JavaScript |

## Build and Run Commands

**Important:** Always use `uv` for package management, running the server, and running Python files. Do not use `pip` directly.

```bash
# Run a Python file
uv run python script.py
```

```bash
# Install dependencies
uv sync

# Run the application (starts FastAPI server on port 8000)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Access points after running:
# - Web Interface: http://localhost:8000
# - API Documentation: http://localhost:8000/docs
```

## Environment Setup

Copy `.env.example` to `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_key_here
```

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials using semantic search and Claude AI.

### Backend Flow (`backend/`)

```
User Query → app.py → rag_system.py → ai_generator.py (Claude API with tools)
                                           ↓
                          search_tools.py ← Claude decides to search
                                           ↓
                          vector_store.py (ChromaDB semantic search)
                                           ↓
                          Returns relevant course chunks → Claude generates answer
```

**Key components:**
- `app.py` - FastAPI server with `/api/query` and `/api/courses` endpoints; serves frontend statically
- `rag_system.py` - Orchestrates all RAG components, processes queries, loads course documents
- `ai_generator.py` - Claude API integration with tool calling (model: claude-sonnet-4-20250514)
- `vector_store.py` - ChromaDB with two collections: `course_catalog` (metadata) and `course_content` (chunks)
- `document_processor.py` - Parses course files, chunks text (800 chars, 100 overlap)
- `search_tools.py` - `CourseSearchTool` for Claude to search courses with optional lesson filtering
- `session_manager.py` - Tracks conversation history per session (max 5 exchanges)
- `config.py` - Centralized settings (embedding model: all-MiniLM-L6-v2)

### Frontend (`frontend/`)

Static HTML/JS/CSS served by FastAPI:
- `index.html` - Two-column layout: sidebar (course stats, suggested questions) + chat area
- `script.js` - Handles chat interactions, session management, markdown rendering via marked.js
- `style.css` - Dark theme with blue accents

### Course Documents (`docs/`)

Text files with specific format:
- Line 1: Course Title
- Line 2: Course Link
- Line 3: Instructor
- Following: Lesson markers (`Lesson N: Title` with optional lesson links) and content

## Key Configuration (backend/config.py)

- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation messages for context
- Vector DB persisted to `backend/chroma_db/`
