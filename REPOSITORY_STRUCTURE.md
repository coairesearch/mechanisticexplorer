# Repository Structure

This document describes the clean, organized structure of the Mechanistic Explorer repository.

## Directory Structure

```
mechanisticexplorer/
├── backend/                    # Backend API server
│   ├── app/                   # Main application package
│   │   ├── __init__.py       # Package initialization
│   │   ├── logit_lens.py     # Real logit lens implementation (nnsight)
│   │   ├── main.py           # FastAPI server and endpoints
│   │   └── models.py         # Pydantic models for API
│   ├── tests/                # Test suite
│   │   ├── __init__.py       # Test package initialization
│   │   ├── run_tests.py      # Test runner script
│   │   ├── test_api_functions.py      # API function tests
│   │   ├── test_api_integration.py    # Full integration tests
│   │   ├── test_logit_lens.py        # LogitLensExtractor tests
│   │   └── test_nnsight_basic.py     # Basic nnsight functionality tests
│   ├── requirements.txt      # Python dependencies
│   ├── run.py               # Development server runner
│   └── start_server.sh      # Server startup script
├── frontend/                 # React frontend application
│   ├── src/                 # Source code
│   │   ├── api/            # API client code
│   │   ├── components/     # React components
│   │   └── types/          # TypeScript type definitions
│   ├── package.json        # Node.js dependencies
│   └── vite.config.ts      # Vite configuration
├── docs/                    # Documentation
│   ├── github-issue-1-logit-lens.md  # Real logit lens implementation spec
│   ├── github-issue-2-caching.md     # Caching system specification
│   ├── implementation-plan.md        # Overall implementation plan
│   └── usage.md                      # Usage documentation
├── playground/              # Development playground scripts
│   └── logit_lens_implementation.py  # Original working example
├── tmp/                     # Temporary development files
│   └── scratchpad.md       # Development notes
├── Claude.md               # Claude Code instructions
└── README.md               # Project overview
```

## Key Files

### Backend Core Files
- **`backend/app/logit_lens.py`**: Main implementation using nnsight for real logit lens extraction
- **`backend/app/main.py`**: FastAPI server with chat and model info endpoints
- **`backend/app/models.py`**: Pydantic models for API request/response structures

### Test Files
- **`backend/tests/run_tests.py`**: Main test runner, run all tests with `python run_tests.py`
- **`backend/tests/test_api_integration.py`**: End-to-end API testing
- **`backend/tests/test_logit_lens.py`**: LogitLensExtractor functionality testing

### Frontend Core Files
- **`frontend/src/components/chat/`**: Chat interface components
- **`frontend/src/components/logitLens/`**: Logit lens visualization components
- **`frontend/src/api/api.ts`**: Backend API client

## Current Implementation Status

✅ **GitHub Issue #1 - Real Logit Lens Extraction**: COMPLETED
- Uses nnsight library for genuine model interpretability
- Extracts layer-by-layer predictions from GPT-2
- API returns real activations (not mock data)

🔄 **GitHub Issue #2 - Caching System**: PENDING
- Documented in `docs/github-issue-2-caching.md`
- Will implement conversation persistence and activation caching

## Running the System

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Tests
```bash
cd backend/tests
python run_tests.py
```

## Clean-up Completed

### Removed Files
- All temporary test files (moved to `tests/` directory)
- Backup implementations (`logit_lens_backup.py`, `main_old.py`, etc.)
- Cloned nnsight repository (was only needed for research)
- Old documentation files

### Organized Structure
- Tests moved to dedicated `tests/` directory
- Only production-ready code in `app/` directory
- Clear separation between frontend and backend
- Comprehensive test suite with single runner script

The repository is now clean, organized, and ready for implementing the caching system (GitHub Issue #2).