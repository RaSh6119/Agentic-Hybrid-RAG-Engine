import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# Configuration
DATA_PATH = "./data"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "tech_ecosystem"

def ingest_vectors():
    # --- Check for API Key ---
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Did you create the .env file?")
        return

    # --- THE FIX: Manual Collection Creation ---
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Reset: Delete if exists
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"🧹 Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass # Ignore if it doesn't exist

    # 2. Create: Define specific schema for OpenAI (1536 dimensions)
    # This bypasses the buggy LangChain initialization
    print(f"🛠️ Creating collection '{COLLECTION_NAME}' with 1536 dimensions...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,  # Standard size for OpenAI text-embedding-ada-002
            distance=models.Distance.COSINE
        )
    )

    # 2. Load Data
    print(f"📂 Scanning {DATA_PATH} for .txt files...")
    documents = []
    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    
    if not txt_files:
        print(f"No files found in {DATA_PATH}. Did you run download_data.py?")
        return

    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            print(f"   - ⚠️ Error loading {os.path.basename(file_path)}: {e}")

    print(f"✅ Successfully loaded {len(documents)} documents.")

    # 3. Split Data
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   - Generated {len(chunks)} text chunks.")

    # 4. Embed & Store (Using add_documents instead of from_documents)
    print(f"🚀 Ingesting into Qdrant...")
    
    try:
        # Initialize the LangChain wrapper with our PRE-MADE client
        vector_store = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=OpenAIEmbeddings()
        )
        # Add the documents to the existing collection
        vector_store.add_documents(chunks)
        
        print("Vector Ingestion Complete! You can now search this data.")
    except Exception as e:
        print(f"Ingestion Error: {e}")

if __name__ == "__main__":
    ingest_vectors()