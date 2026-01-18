from core.router import route_question
from core.retriever import search_vector, search_graph

def run_test():
    # Test Questions
    questions = [
        "What is the history of Microsoft?",        # Should be VECTOR
        "Who is the CEO of Tesla?",                 # Should be GRAPH
        "Does Meta own Instagram?",                 # Should be GRAPH
        "Summarize the risks of AI development."    # Should be VECTOR
    ]

    print("🧠 STARTING BRAIN TEST...\n")

    for q in questions:
        print(f"\n--- Question: {q} ---")
        
        # 1. Ask the Brain where to go
        destination = route_question(q)
        print(f"👉 DECISION: {destination.upper()}")
        
        # 2. Execute the decision
        if destination == "vector_store":
            print("   Running Vector Search...")
            # We just print the first 100 chars to prove it retrieved something
            result = search_vector(q)
            print(f"   📄 Result Snippet: {result[:100]}...")
            
        elif destination == "graph_store":
            print("   Running Graph Search...")
            result = search_graph(q)
            print(f"   🕸️ Result: {result}")

if __name__ == "__main__":
    run_test()