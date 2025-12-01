"""
Knowledge Base to ChromaDB: Ingest Prolog KB into vector database for RAG.

This module handles the ingestion of Prolog knowledge bases into ChromaDB,
enabling semantic retrieval of relevant facts and rules for each query.

Key Features:
- Convert Prolog facts/rules to natural language descriptions
- Embed and store in ChromaDB with metadata
- Semantic search over knowledge base
- Persistent storage across sessions

RAG Pipeline:
1. Parse Prolog KB → Facts and Rules
2. Generate natural language descriptions
3. Store in ChromaDB (auto-embedding)
4. Query with natural language → Retrieve relevant entries

Critical for:
- Semantic retrieval instead of loading entire KB
- Scalability to large knowledge bases
- Natural language querying
"""

import os
import sys
from typing import List, Dict
import chromadb
from chromadb.config import Settings

# Import prolog parser
from prolog_parser import parse_prolog_file


def create_fact_document(fact: object) -> Dict[str, str]:
    """
    Create a document for a fact with metadata.
    
    Args:
        fact: Fact object
        
    Returns:
        Document dictionary
    """
    fact_str = str(fact)
    
    # Create natural language description
    predicate = fact.predicate
    args = fact.args if hasattr(fact, 'args') else []
    
    # Generate human-readable description
    if predicate == "park_worker":
        description = f"{args[0]} is a park worker"
    elif predicate == "boss":
        description = f"{args[0]} is a boss"
    elif predicate == "park_manager":
        description = f"{args[0]} is a park manager"
    elif predicate == "character_type":
        description = f"{args[0]} is a {args[1]}"
    elif predicate == "friends":
        description = f"{args[0]} and {args[1]} are friends"
    elif predicate == "reports_to":
        description = f"{args[0]} reports to {args[1]}"
    else:
        description = fact_str
    
    return {
        "text": description,
        "prolog": fact_str,
        "predicate": predicate,
        "type": "fact"
    }


def create_rule_document(rule: object) -> Dict[str, str]:
    """
    Create a document for a rule with metadata.
    
    Args:
        rule: Rule object
        
    Returns:
        Document dictionary
    """
    rule_str = str(rule)
    
    # Create natural language description
    conclusion = rule.conclusion
    predicate = conclusion.predicate
    
    # Generate human-readable description
    if predicate == "in_charge_of":
        description = "Someone is in charge of another person if that person reports to them"
    elif predicate == "work_together":
        description = "Two people work together if they are both park workers and different people"
    elif predicate == "has_authority":
        description = "Someone has authority if they are a boss or park manager"
    elif predicate == "is_subordinate":
        description = "Someone is a subordinate if they report to someone"
    else:
        description = f"Rule: {rule_str}"
    
    return {
        "text": description,
        "prolog": rule_str,
        "predicate": predicate,
        "type": "rule"
    }


def ingest_kb_to_chromadb(kb_path: str, collection_name: str = "regular_show_kb") -> chromadb.Collection:
    """
    Ingest Regular Show knowledge base into ChromaDB.
    
    Args:
        kb_path: Path to Prolog KB file
        collection_name: Name of ChromaDB collection
        
    Returns:
        ChromaDB collection
    """
    print(f"Parsing KB from {kb_path}...")
    facts, rules = parse_prolog_file(kb_path)
    print(f"✓ Parsed {len(facts)} facts and {len(rules)} rules")
    
    # Initialize ChromaDB client (persistent)
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Regular Show Knowledge Base"}
    )
    print(f"✓ Created collection: {collection_name}")
    
    # Prepare documents
    documents = []
    metadatas = []
    ids = []
    
    # Add facts
    for i, fact in enumerate(facts):
        doc = create_fact_document(fact)
        documents.append(doc["text"])
        metadatas.append({
            "prolog": doc["prolog"],
            "predicate": doc["predicate"],
            "type": doc["type"]
        })
        ids.append(f"fact_{i}")
    
    # Add rules
    for i, rule in enumerate(rules):
        doc = create_rule_document(rule)
        documents.append(doc["text"])
        metadatas.append({
            "prolog": doc["prolog"],
            "predicate": doc["predicate"],
            "type": doc["type"]
        })
        ids.append(f"rule_{i}")
    
    # Add to collection
    print(f"Ingesting {len(documents)} documents into ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✓ Ingested {len(documents)} documents")
    
    return collection


def query_kb(query: str, collection_name: str = "regular_show_kb", n_results: int = 10) -> List[Dict]:
    """
    Query the knowledge base from ChromaDB.
    
    Args:
        query: Natural language query
        collection_name: Name of ChromaDB collection
        n_results: Number of results to return
        
    Returns:
        List of relevant documents
    """
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    documents = []
    for i in range(len(results['documents'][0])):
        documents.append({
            "text": results['documents'][0][i],
            "prolog": results['metadatas'][0][i]['prolog'],
            "predicate": results['metadatas'][0][i]['predicate'],
            "type": results['metadatas'][0][i]['type']
        })
    
    return documents


if __name__ == "__main__":
    # Ingest the Regular Show KB
    import sys
    
    kb_path = "../task_4/regular_show_kb.pl"
    if len(sys.argv) > 1:
        kb_path = sys.argv[1]
    
    if not os.path.exists(kb_path):
        print(f"Error: KB file not found at {kb_path}")
        print("Usage: python kb_to_chromadb.py [path_to_prolog_file]")
        sys.exit(1)
    
    print("Ingesting knowledge base...")
    collection = ingest_kb_to_chromadb(kb_path)
    print("✓ Ingestion complete!")
    
    # Test a query
    test_query = "Who is in charge of Mordecai?"
    print(f"\nTest query: {test_query}")
    results = query_kb(test_query, n_results=5)
    print(f"Retrieved {len(results)} relevant documents")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. [{doc['type']}] {doc['prolog']}")

