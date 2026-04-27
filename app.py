from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama


CHROMA_PATH = "chroma_db"
MODEL_NAME = "qwen2:0.5b"   


app = FastAPI(title="RAG Chroma API", version="1.0")


try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

except Exception as e:
    print(f"❌ Error loading DB: {e}")
    db = None

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running 🚀",
        "endpoints": {
            "ask": "POST /ask",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def generate_answer(query, docs):
    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,   
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"



@app.post("/ask")
def ask(request: QueryRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not loaded")

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Semantic Search
        docs = db.similarity_search(query, k=3)

        # Generate Answer
        answer = generate_answer(query, docs)

        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content_preview": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))