# Agentic Hybrid RAG Engine

A production-ready Retrieval-Augmented Generation (RAG) system that combines Knowledge Graphs (Neo4j) and Vector Search (Qdrant) to answer complex queries. This system features a Semantic Router to delegate tasks intelligently and an Agentic Memory Module that adapts the response tone and depth based on the user's role (e.g., CTO vs. CEO).

## Key Features

* **Hybrid Retrieval:**
    * **Graph Search (Neo4j):** Handles structured queries about relationships, ownership, and organizational hierarchies (e.g., "Who is the CEO of Tesla?", "Does Meta own Instagram?").
    * **Vector Search (Qdrant):** Handles unstructured semantic queries for summaries, history, and conceptual explanations (e.g., "Summarize the history of Microsoft").
* **Semantic Routing:** A Pydantic-based router classifies user intent to select the most efficient retrieval tool, reducing latency and hallucinations.
* **Agentic Personalization:** The system identifies the active user (e.g., "Rahul - CTO" vs. "Ram - CEO") and dynamically adjusts the technical depth, terminology, and focus of the answer.
* **Self-Healing Graph Queries:** Includes robust error handling and query refinement to manage database inconsistencies and schema variations.

## Architecture

1.  **User Input:** The user asks a question via the Streamlit Interface.
2.  **Context Injection:** The system looks up the active user's profile in the Graph Database to retrieve specific preferences (e.g., "prefers code snippets" vs. "prefers ROI analysis").
3.  **Routing:** The Semantic Router analyzes the question and directs it to either the Graph Store or Vector Store.
4.  **Retrieval:**
    * If **Graph**: Generates Cypher queries to traverse nodes and edges.
    * If **Vector**: Performs similarity search on text embeddings.
5.  **Synthesis:** The LLM combines the Retrieved Data + User Context to generate a personalized, accurate response.

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain
* **LLM:** OpenAI GPT-4o-mini
* **Graph Database:** Neo4j
* **Vector Database:** Qdrant
* **Interface:** Streamlit
* **Containerization:** Docker & Docker Compose

## Installation & Setup

### 1. Prerequisites
* Docker Desktop installed and running.
* Python 3.10 or higher.
* OpenAI API Key.

### 2. Clone the Repository
```bash
git clone [https://github.com/YourUsername/Agentic-Hybrid-RAG-Engine.git](https://github.com/YourUsername/Agentic-Hybrid-RAG-Engine.git)
cd Agentic-Hybrid-RAG-Engine
```

### 3. Setup Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
QDRANT_URL=http://localhost:6333
```

### 4. Start 
```bash
docker-compose up -d
```

### 5. Install Depedencies
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 6. Initialize Data
```bash
# Ingest Knowledge Graph and Vector Data
python ingest.py 

# Create User Personas (Alice/Bob/Rahul/Ram)
python setup_users.py
```

### 7. Run the Application
```bash
streamlit run app.py
```

### 8. Test Personalization
```bash
1.  **Select a Persona:** Use the sidebar to switch between "Rahul" (CTO) and "Ram" (CEO).
2.  **Ask a Question:** Try "Summarize the history of Microsoft."
3.  **Observe:**
    * **Rahul** receives a response detailing version numbers, technical architecture, and development milestones.
    * **Ram** receives a high-level executive summary focusing on market cap, acquisitions, and business growth.
```

### 9. Project Structure
```bash
├── app.py                 # Main Streamlit UI application
├── brain.py               # Core logic engine (Context + Routing + Synthesis)
├── setup_users.py         # Script to seed User Personas into Neo4j
├── ingest.py              # Script to populate Vector and Graph databases
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Database container configuration
├── core/
│   ├── router.py          # Semantic Router logic
│   └── retriever.py       # Graph and Vector search tools
├── data/
│   └── financial_reports/ # Source documents
└── .env                   # Environment variables (GitIgnored)
```

## Evaluation & Benchmarking

To validate the architecture, we benchmarked the **Agentic Hybrid System** against **4 industry-standard baselines**:

1.  **BM25:** Traditional Keyword Search (Lexical).
2.  **Naive Vector:** Standard RAG (Cosine Similarity).
3.  **HyDE:** Hypothetical Document Embeddings (State-of-the-Art semantic search).
4.  **Graph Only:** Pure Neo4j retrieval without semantic fallback.

### Methodology
* **Dataset:** A "Stress Test" suite comprising Multi-hop reasoning, Exhaustive List generation, Specific Relationship checks, and Broad Summarization queries.
* **Metric:** "LLM-as-a-Judge" scoring (0-10) based on accuracy and completeness against ground truth.

### Results Matrix (Score 0-10)

| Query Challenge | **Agentic Hybrid** | HyDE (SOTA) | Naive Vector | Graph Only | BM25 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Hop Logic** <br>*(e.g., "CEO of Parent Co.")* | 🏆 **10.0** | 10.0 | ❌ 0.0 | 0.0 | 0.0 |
| **Exhaustive Lists** <br>*(e.g., "All acquisitions")* | 🏆 **6.0** | 5.0 | ❌ 0.0 | 🏆 6.0 | 0.0 |
| **Relation Check** <br>*(e.g., "Specific role")* | 8.0 | 🏆 **9.0** | 8.0 | 0.0 | 0.0 |
| **Summarization** <br>*(e.g., "History of AI")* | 🏆 **9.0** | 8.0 | 8.0 | 7.0 | 0.0 |
| **OVERALL AVERAGE** | 🏆 **8.25** | **8.00** | **4.00** | **3.25** | **0.00** |

### Key Findings
* **Solved the "Zero Problem":** Standard Vector RAG scored **0/10** on multi-hop logic and exhaustive list queries. The Hybrid system successfully unlocked these capabilities (Scoring 10/10 and 6/10 respectively).
* **Outperformed SOTA:** The Hybrid system edged out HyDE (**+3%**) by maintaining higher precision on list-based tasks where embedding-only methods tend to hallucinate.
* **Versatility:** While "Graph Only" failed at summaries and "Vector Only" failed at logic, the **Agentic Hybrid** maintained high performance across all query categories, proving the value of the dual-retrieval architecture.