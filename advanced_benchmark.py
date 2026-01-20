import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from core.retriever import search_vector, search_graph, get_user_context
from brain import ask_brain
import qdrant_client

# --- CONFIGURATION ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings()

# --- 1. SETUP BM25 (Keyword Baseline) ---
# We need to fetch all chunks from Qdrant to build the BM25 index locally
# In a real production system, you'd use a search engine like Elasticsearch/Solr for this.
print("⏳ Building BM25 Index (Fetching chunks)...")
client = qdrant_client.QdrantClient(url="http://localhost:6333")
# Fetch 1000 points (assuming small dataset for demo)
records = client.scroll(
    collection_name="tech_ecosystem", 
    limit=1000, 
    with_payload=True
)[0]

documents = [r.payload['page_content'] for r in records if 'page_content' in r.payload]
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
print(f"✅ BM25 Index ready with {len(documents)} documents.")

# --- HELPER FUNCTIONS FOR BASELINES ---

def run_bm25(question):
    """Baseline 1: Old-school Keyword Search"""
    tokenized_query = question.split(" ")
    # Get top 3 docs
    top_docs = bm25.get_top_n(tokenized_query, documents, n=3)
    context = "\n".join(top_docs)
    prompt = f"Answer strictly based on this context:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

def run_naive_vector(question):
    """Baseline 2: Standard Vector Search"""
    context = search_vector(question)
    prompt = f"Answer strictly based on this context:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

def run_hyde(question):
    """Baseline 3: HyDE (Hypothetical Document Embeddings)"""
    # Step 1: Hallucinate a 'fake' perfect answer
    hyde_prompt = f"Write a hypothetical passage that answers the question: {question}"
    hypothetical_answer = llm.invoke(hyde_prompt).content
    
    # Step 2: Search Vector store using the FAKE answer as the query
    # (This matches 'intent' better than the raw question)
    context = search_vector(hypothetical_answer) 
    
    prompt = f"Answer strictly based on this context:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

def run_graph_only(question):
    """Baseline 4: Graph Only (No Vector Fallback)"""
    context = search_graph(question)
    if "I don't know" in str(context) or not context:
        return "I could not find an answer in the Knowledge Graph."
    
    prompt = f"Answer strictly based on this context:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt).content

def run_agentic_hybrid(question):
    """Your System: The Hybrid Agent"""
    # We use 'Ram' (CEO) as the default persona for consistency
    return ask_brain(question, user_id="Ram")

# --- EVALUATION LOGIC ---

def evaluate_answer(question, answer, ground_truth):
    """LLM-as-a-Judge: Grades 0-10"""
    grader_prompt = f"""
    You are a strict evaluator.
    
    QUESTION: {question}
    GROUND TRUTH: {ground_truth}
    SYSTEM ANSWER: {answer}
    
    Grade the SYSTEM ANSWER from 0 to 10 based on accuracy and completeness.
    If the answer says "I don't know" or is wrong, give 0.
    
    Return ONLY the integer.
    """
    try:
        score = evaluator_llm.invoke(grader_prompt).content
        return int(''.join(filter(str.isdigit, score)))
    except:
        return 0

# --- DATASET (The "Stress Test") ---
test_set = [
    {"q": "Who is the CEO of the parent company of Instagram?", "truth": "Mark Zuckerberg (CEO of Meta)", "type": "Multi-Hop"},
    {"q": "List all companies that have been acquired by Tesla.", "truth": "SolarCity, Maxwell, Grohmann, Perbix, Hibar, Riviera Tool, Compass Automation.", "type": "List"},
    {"q": "Summarize the history of Microsoft.", "truth": "Founded 1975 by Gates/Allen. Created MS-DOS, Windows. IPO 1986. Leaders: Gates -> Ballmer -> Nadella. Now Cloud/AI giant.", "type": "Summary"},
    {"q": "What is the specific relationship between Elon Musk and SolarCity?", "truth": "Musk was Chairman and cousin of founders (Rive brothers). Tesla acquired it.", "type": "Relation"},
]

# --- MAIN BENCHMARK LOOP ---
results = []

print("\n🚀 STARTING ADVANCED BENCHMARK...\n")

for item in test_set:
    q = item['q']
    truth = item['truth']
    q_type = item['type']
    
    print(f"🧪 Testing ({q_type}): {q}")
    
    # Run all models
    answers = {
        "BM25": run_bm25(q),
        "Naive Vector": run_naive_vector(q),
        "HyDE": run_hyde(q),
        "Graph Only": run_graph_only(q),
        "Agentic Hybrid": run_agentic_hybrid(q)
    }
    
    # Grade all models
    scores = {}
    for model_name, ans in answers.items():
        score = evaluate_answer(q, ans, truth)
        scores[model_name] = score
        
        # Save detailed row
        results.append({
            "Question": q,
            "Type": q_type,
            "Model": model_name,
            "Score": score,
            "Answer": ans
        })

print("\n✅ Evaluation Complete.")

# --- REPORTING ---
df = pd.DataFrame(results)

# Pivot table for clean Resume/README view
pivot_df = df.pivot(index="Type", columns="Model", values="Score")
# Reorder columns to show progression
pivot_df = pivot_df[["BM25", "Naive Vector", "HyDE", "Graph Only", "Agentic Hybrid"]]

print("\n🏆 FINAL SCORECARD (0-10)")
print("=========================")
print(pivot_df)

# Calculate Average Win
avg_scores = df.groupby("Model")["Score"].mean().sort_values(ascending=False)
print("\n📈 OVERALL RANKING (Avg Score):")
print(avg_scores)

# Save
df.to_csv("advanced_benchmark_details.csv", index=False)
pivot_df.to_markdown("advanced_benchmark_summary.md")
print("\n📄 Report saved to 'advanced_benchmark_summary.md'")