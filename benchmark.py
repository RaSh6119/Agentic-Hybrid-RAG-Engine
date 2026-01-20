import pandas as pd
from core.retriever import search_vector
from brain import ask_brain
from langchain_openai import ChatOpenAI

# --- Configuration ---
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Define the "Golden Dataset"
# Questions that test different parts of your system
test_set = [
    # --- TEST 1: The "Multi-Hop" Query ---
    # Why it's hard: Vector search finds "Instagram" docs or "Meta" docs, 
    # but rarely a single sentence linking Instagram -> Meta -> Zuckerberg clearly.
    # Graph approach: (Instagram)<-[:OWNS]-(Meta)-[:CEO]->(Person)
    {
        "question": "Who is the CEO of the parent company of Instagram?",
        "ground_truth": "Mark Zuckerberg (CEO of Meta, which owns Instagram).",
        "type": "Multi-Hop Logic"
    },

    # --- TEST 2: The "Exhaustive List" Query ---
    # Why it's hard: Vector search retrieves top-3 chunks. If Tesla bought 5 companies,
    # the chunks might only mention 1 or 2. 
    # Graph approach: MATCH (Tesla)-[:ACQUIRED]->(c) RETURN c (Finds ALL).
    {
        "question": "List all companies that have been acquired by Tesla.",
        "ground_truth": "SolarCity, Maxwell Technologies, Grohmann Engineering, Perbix, Hibar Systems, Riviera Tool, Compass Automation.",
        "type": "Exhaustive List"
    },

    # --- TEST 3: The "Specific Relationship" Query ---
    # Why it's hard: Vector search sees "Musk" and "SolarCity" near each other and guesses "Founder".
    # Graph approach: Finds the exact edge (e.g., "Family Connection" or "Board Member").
    {
        "question": "What is the exact relationship between Elon Musk and SolarCity?",
        "ground_truth": "Elon Musk was the Chairman of SolarCity and is the cousin of its founders (Lyndon and Peter Rive). Tesla later acquired it.",
        "type": "Specific Relationship"
    },

    # --- TEST 4: The "Negative/Filter" Query ---
    # Why it's hard: Vector search ignores the word "NOT". It will return results about Microsoft owning things.
    # Graph approach: Can verify absence of edges (or we rely on Vector with strict persona constraints).
    {
        "question": "Did Microsoft acquire Tesla? Answer Yes or No.",
        "ground_truth": "No.",
        "type": "Fact Verification"
    }
]

def get_baseline_answer(question):
    """Simulates a 'Dumb' RAG (Vector Only, No Persona, No Routing)"""
    # Force Vector Search
    context = search_vector(question)
    # Simple generation without the fancy brain logic or persona
    prompt = f"Answer this based only on context: {context}\n\nQuestion: {question}"
    try:
        return evaluator_llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {e}"

def evaluate_answer(question, answer, ground_truth):
    """Uses an LLM to grade the answer from 0-10."""
    grader_prompt = f"""
    You are a strict teacher grading an exam.
    
    QUESTION: {question}
    GROUND TRUTH: {ground_truth}
    STUDENT ANSWER: {answer}
    
    Grade the STUDENT ANSWER from 0 to 10 based on:
    1. Accuracy (Does it match the ground truth?)
    2. Completeness (Did it miss key details?)
    
    Return ONLY the number (e.g., 8).
    """
    try:
        score = evaluator_llm.invoke(grader_prompt).content
        # Extract number safely (handles responses like "The score is 8")
        return int(''.join(filter(str.isdigit, score))) 
    except:
        return 0

# --- Main Execution ---
results = []

print("📊 STARTING BENCHMARKING...\n")

for item in test_set:
    q = item["question"]
    print(f"   🧪 Testing: {q}...")
    
    # 1. Run Baseline (Naive RAG)
    baseline_ans = get_baseline_answer(q)
    baseline_score = evaluate_answer(q, baseline_ans, item["ground_truth"])
    
    # 2. Run Agentic Brain (Your System)
    # We use 'Ram' (CEO) persona for a balanced, high-level answer
    agentic_ans = ask_brain(q, user_id="Ram")
    agentic_score = evaluate_answer(q, agentic_ans, item["ground_truth"])
    
    results.append({
        "Question": q,
        "Type": item["type"],
        "Baseline Answer": baseline_ans,   # Save the actual text
        "Agentic Answer": agentic_ans,     # Save the actual text
        "Baseline Score": baseline_score,
        "Agentic Score": agentic_score,
        "Winner": "Agentic" if agentic_score > baseline_score else "Tie/Baseline"
    })

# --- Reporting ---
df = pd.DataFrame(results)

# Print Summary to Console
print("\n\n🏆 FINAL RESULTS REPORT")
print("========================")
print(df[["Type", "Baseline Score", "Agentic Score", "Winner"]])

avg_baseline = df["Baseline Score"].mean()
avg_agentic = df["Agentic Score"].mean()

print(f"\n📈 Average Baseline Score: {avg_baseline:.1f}/10")
print(f"🚀 Average Agentic Score:  {avg_agentic:.1f}/10")

# Save detailed results to CSV (for inspection)
df.to_csv("benchmark_results.csv", index=False)
print("\n✅ Detailed results saved to 'benchmark_results.csv'")

# Save summary to Markdown (for README)
df[["Question", "Type", "Baseline Score", "Agentic Score", "Winner"]].to_markdown("benchmark_summary.md", index=False)
print("✅ Summary table saved to 'benchmark_summary.md'")