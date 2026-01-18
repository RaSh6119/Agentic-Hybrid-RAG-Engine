from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# Connect in "Blind Mode" to bypass the crash
graph = Neo4jGraph(
    url='bolt://localhost:7687', 
    username='neo4j', 
    password='password123',
    enhanced_schema=False,
    refresh_schema=False 
)

print("🔍 Checking for remaining duplicates...")

# Count how many pairs of nodes have more than 1 relationship of the same type
check_query = """
MATCH (a)-[r]->(b)
WITH a, b, type(r) AS t, count(r) AS count
WHERE count > 1
RETURN a.id, b.id, t, count
"""

res = graph.query(check_query)

if not res:
    print("✅ CONFIRMED: No duplicate relationships found. The database is clean.")
else:
    print(f"❌ Still found {len(res)} duplicates! Here is one: {res[0]}")