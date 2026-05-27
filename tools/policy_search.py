from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the embedding function and vector store
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
except Exception as e:
    vector_store = None
    print(f"Failed to load Chroma DB: {e}")

@tool
def search_policy_docs(query: str) -> list:
    """RAG search over the uploaded policy PDF. Use this to find information about waiting periods, general rules, exclusions, late claim submission, etc. Returns top relevant chunks."""
    if vector_store is None:
        return [{"error": "Vector store not initialized. Please ensure a policy PDF has been uploaded and processed."}]
        
    try:
        results = vector_store.similarity_search(query, k=3)
        return [{"chunk": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        return [{"error": str(e)}]
