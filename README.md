# RAG System with PostgreSQL Vector Database

A Retrieval-Augmented Generation (RAG) system built with FastAPI and PostgreSQL with pgvector extension for efficient vector similarity search.

## 🌟 Features

- **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
- **Multiple Input Sources**:
  - Text documents
  - PDF files
  - Web scraping
- **FastAPI Backend**: High-performance asynchronous API
- **Modular Architecture**: Easily extensible for different embedding models and LLMs

## 🛠️ Architecture

The system consists of several components:

1. **Embedding Model**: Converts text into vector embeddings
2. **Vector Database**: Stores and retrieves embeddings efficiently
3. **Document Loaders**: Process different document types (PDF, web pages)
4. **RAG Pipeline**: Orchestrates the retrieval and generation process
5. **API Layer**: Exposes functionality through REST endpoints

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- API keys for your LLM provider (if using OpenAI or similar)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if using Pipenv:
   ```bash
   pipenv install
   ```

3. Set up environment variables in `.env` file:
   ```
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Start the application:
   ```bash
   uvicorn main:app --reload
   ```

## 📚 API Endpoints

### Query Endpoint
```
POST /query
```
Send a question to the RAG system.

### Ingestion Endpoints
```
POST /texts      # Ingest plain text
POST /pdf        # Ingest PDF files
POST /web        # Scrape and ingest web content
```

### Health Check
```
GET /health
```

## 🧩 Components

### Database Client
The `PostgresVdbClient` class handles all interactions with the PostgreSQL vector database, including:
- Creating tables with vector columns
- Batch inserting embeddings
- Similarity search
- Connection pooling

### Embedding Models
The system uses a modular approach to embedding models, with `BasePgVectorEmbeddingModel` providing common functionality.

### Document Processors
- `PDFProcessor`: Extracts and processes text from PDF files
- `WebScraper`: Scrapes content from web pages

### RAG Pipeline
The `RagPipeline` class orchestrates the entire RAG process:
- Document ingestion and chunking
- Embedding generation
- Retrieval of relevant context
- Query processing with LLM

## 🔧 Configuration

The application uses Pydantic's `BaseSettings` for configuration management, loading values from environment variables:

```
# PostgreSQL
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_MIN_CONN=1
DB_MAX_CONN=10

# OpenAI
OPENAI_API_KEY=sk-your-api-key-here

# App Settings
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=True

# FastAPI
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_RELOAD=True

# LLM Settings
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.7

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_DIM=1536

# Web Scraping
WEB_TIMEOUT=10
MAX_URLS_PER_REQUEST=10
```

## 🧪 Development

### Running Tests
```bash
pytest
```

### Local Development
```bash
uvicorn main:app --reload --port 8000
```

## 📝 License

[MIT License](LICENSE)
