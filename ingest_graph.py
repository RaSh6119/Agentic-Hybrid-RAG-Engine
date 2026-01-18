import os
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# Configuration
DATA_PATH = "./data"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
MODEL_NAME = "gpt-4o-mini" 

# 🛑 SAFETY LIMIT: Set to None to process EVERYTHING.
CHUNK_LIMIT = None 
MAX_WORKERS = 5  # Number of parallel requests (Don't go too high or you hit Rate Limits)

def process_batch(transformer, batch, batch_index):
    """Helper function to process a single batch of text."""
    try:
        print(f"   ⏳ Starting batch {batch_index}...")
        graph_documents = transformer.convert_to_graph_documents(batch)
        print(f"   ✅ Batch {batch_index} processed ({len(graph_documents)} docs).")
        return graph_documents
    except Exception as e:
        print(f"   ❌ Error in batch {batch_index}: {e}")
        return []

def ingest_graph():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found.")
        return

    print(f"🔄 Connecting to Neo4j at {NEO4J_URI}...")
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    except Exception as e:
        print(f"❌ Neo4j Connection Failed: {e}")
        return

    # 2. Load Data
    print(f"📂 Scanning {DATA_PATH} for .txt files...")
    documents = []
    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception:
            pass # Skip errors for speed

    # 3. Split Data
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    if CHUNK_LIMIT:
        chunks = chunks[:CHUNK_LIMIT]
    
    print(f"   - Total chunks to process: {len(chunks)}")

    # 4. Initialize Transformer
    llm = ChatOpenAI(temperature=0, model=MODEL_NAME)
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 5. Parallel Extraction
    print(f"🚀 Starting Parallel Extraction with {MAX_WORKERS} workers...")
    
    # We group chunks into mini-batches to reduce overhead
    BATCH_SIZE = 5 
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(process_batch, llm_transformer, batch, i): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they finish
        for future in as_completed(future_to_batch):
            graph_docs = future.result()
            if graph_docs:
                results.extend(graph_docs)
                # Optional: Write to DB immediately to save progress?
                # For safety, we can write per-batch, but let's just collect all for simplicity
                try:
                    graph.add_graph_documents(graph_docs)
                    print("      💾 Saved batch to Neo4j")
                except Exception as e:
                    print(f"      ❌ DB Write Error: {e}")

    print("✨ Graph Ingestion Complete!")

if __name__ == "__main__":
    ingest_graph()