# AI-RAG-assistant

## Description
This project is an AI-powered assistant for business documents. It allows users to upload documents and ask questions based strictly on the document content.

## Features
- PDF/DOCX document ingestion
- Chunking and embedding using Sentence Transformers
- Vector storage using Chroma DB
- Semantic search
- LLM-based answer generation using Ollama
- FastAPI backend with Swagger UI

## How to Run

### 1. Install dependencies
- pip install -r requirements.txt

### 2. Create vector database
- python ingestion_pipeline.py

### 3. Run API
- uvicorn app:app --reload

### 4. Access API Documentation (Swagger UI)
- After running the FastAPI server locally, the API documentation can be accessed via Swagger UI, which provides an interactive interface to test the endpoints such as document-based question answering.
- The `/ask` endpoint can be tested directly from this interface by providing natural language queries related to uploaded business documents.
