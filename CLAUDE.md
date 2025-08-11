# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start Commands

**Run the application:**
```bash
chmod +x run.sh && ./run.sh
```

**Manual startup:**
```bash
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync --group dev  # Includes development tools
```

**Environment setup:**
Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`

**Code Quality Tools:**
```bash
make quality      # Run all quality checks (format, lint, test)
make format       # Format code with black and isort
make lint         # Run linting tools (flake8, mypy)  
make test         # Run all tests
make install      # Install all dependencies
```

## Project Guidelines

- Always use uv to run the server, do not use pip directly
- Make sure to use uv to manage all dependencies
- Run `make quality` before committing code to ensure formatting and linting standards
- Use `make format` to automatically format code with Black and organize imports with isort

## Architecture Overview

This is a full-stack RAG (Retrieval-Augmented Generation) chatbot system for querying course materials. The system uses ChromaDB for vector storage, Anthropic's Claude for AI responses, and FastAPI for the backend API.

### Core Components

**Frontend (`frontend/`):**
- `script.js`: Main chat interface, handles user input and API communication
- `index.html`: Single-page chat application with sidebar for course stats
- Sends POST requests to `/api/query` with query and optional session_id

**Backend Architecture (`backend/`):**
- `app.py`: FastAPI server with two main endpoints: `/api/query` (chat) and `/api/courses` (stats)
- `rag_system.py`: Central orchestrator coordinating all components
- `ai_generator.py`: Anthropic Claude API integration with tool calling capabilities
- `search_tools.py`: Tool-based search system using CourseSearchTool for semantic queries
- `vector_store.py`: ChromaDB wrapper with dual collections (course_catalog + course_content)
- `document_processor.py`: Parses course documents with structured format (title/instructor/lessons)
- `session_manager.py`: Conversation history and context management
- `models.py`: Pydantic models (Course, Lesson, CourseChunk)
- `config.py`: Configuration with environment variables

### Query Processing Flow

1. **User Input** → Frontend captures query via `sendMessage()`
2. **API Request** → POST to `/api/query` with QueryRequest model
3. **RAG Orchestration** → `rag_system.query()` manages session and coordinates components
4. **AI Processing** → Claude analyzes query and decides whether to use search tools
5. **Vector Search** → If needed, CourseSearchTool performs semantic search on ChromaDB
6. **Response Generation** → AI synthesizes search results into coherent answer
7. **Session Update** → Conversation history stored for context
8. **Frontend Display** → JSON response with answer and sources displayed in chat

### Data Architecture

**Document Format:** Course documents follow structured format:
```
Course Title: [title]
Course Link: [url] 
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [url]
[lesson content...]
```

**Vector Storage:**
- `course_catalog` collection: Course metadata for name resolution
- `course_content` collection: Text chunks with course/lesson context
- Chunks include contextual prefixes: "Course [title] Lesson [num] content: [chunk]"

**Models:**
- `Course`: Contains title (unique ID), instructor, lessons list
- `Lesson`: lesson_number, title, optional lesson_link  
- `CourseChunk`: content, course_title, lesson_number, chunk_index

### Tool System

The AI uses a tool-based architecture where Claude can call `search_course_content` tool:
- Supports semantic course name matching (partial names work)
- Optional lesson number filtering
- Returns formatted results with course/lesson context
- Sources tracked automatically for frontend display

### Configuration

Key settings in `config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks  
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation messages remembered
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"`
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"`

### Development Notes

**Adding Course Documents:** Place `.txt`, `.pdf`, or `.docx` files in `docs/` folder. They're loaded automatically on startup via `startup_event()` in `app.py`.

**ChromaDB Storage:** Persistent storage in `./chroma_db` directory with two collections for metadata and content separation.

**Session Management:** Sessions auto-created on first query, conversation history maintained with configurable depth for AI context.

**Error Handling:** Graceful degradation - if vector search fails, AI falls back to general knowledge. Frontend shows loading states and error messages.

**Tool Execution:** AI can execute one search per query maximum. Tool results are synthesized into natural responses without meta-commentary about the search process.

### Code Quality Tools

**Formatting and Linting:**
- **Black**: Automatic code formatting with 88-character line length
- **isort**: Import statement organization and sorting
- **flake8**: Style and syntax checking
- **mypy**: Static type checking for better code reliability

**Development Scripts (`scripts/`):**
- `format.py`: Runs Black and isort for code formatting
- `lint.py`: Runs flake8 and mypy for quality checks
- `test.py`: Executes pytest test suite
- `quality.py`: Runs all quality tools in sequence

**Configuration:**
- All tool configurations are in `pyproject.toml`
- Black configured for 88-character lines, Python 3.13 compatibility
- isort configured to work with Black formatting
- mypy configured with strict type checking options

**Usage:**
Use `make quality` before commits to ensure code meets project standards. The Makefile provides convenient shortcuts for all development tasks.