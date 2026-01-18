from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# Connect to Graph (Standard Mode)
graph = Neo4jGraph(
    url='bolt://localhost:7687', 
    username='neo4j', 
    password='password123',
    enhanced_schema=False,
    refresh_schema=False
)

print("🧹 STARTING DATABASE CLEANUP...")

# 1. Deduplicate Relationships
# This query finds where (A)-[r]->(B) has duplicates and keeps only one.
dedup_query = """
MATCH (a)-[r]->(b)
WITH a, b, type(r) AS t, collect(r) AS rels
WHERE size(rels) > 1
FOREACH (r IN tail(rels) | DELETE r)
"""

print("   - Removing duplicate relationships...")
try:
    graph.query(dedup_query)
    print("   ✅ Duplicates removed.")
except Exception as e:
    print(f"   ❌ Error removing duplicates: {e}")

# 2. Verify Fix by refreshing schema
print("   - Verifying Schema Integrity...")
try:
    graph.refresh_schema()
    print("   ✨ SUCCESS! The database schema is now valid and clean.")
except Exception as e:
    print(f"   ⚠️ Schema still has issues: {e}")