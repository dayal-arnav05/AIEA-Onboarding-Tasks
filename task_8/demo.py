#!/usr/bin/env python3
"""
Demo: LangChain + RAG + Backward Chaining Reasoning System

This script demonstrates the complete neuro-symbolic AI pipeline that combines:
- Neural: LLM-powered natural language understanding (LangChain)
- Symbolic: Formal logical inference (Backward Chaining)
- Hybrid: Semantic knowledge retrieval (RAG with ChromaDB)

Pipeline Stages:
1. Setup: Ingest Prolog KB into ChromaDB (one-time)
2. Query: Natural language question
3. RAG: Retrieve relevant facts/rules semantically
4. Parse: Convert question to formal query
5. Infer: Backward chaining with variable unification
6. Output: TRUE/FALSE with logical deduction trace

Example Output:
    Question: Is Benson in charge of Mordecai?
    Result: TRUE
    
    Trace:
      Goal: in_charge_of(benson, mordecai)
        Trying rule: in_charge_of(?X,?Y) :- reports_to(?Y,?X)
        Goal: reports_to(mordecai, benson)
          ✓ Matched fact: reports_to(mordecai, benson)
        ✓ Rule succeeded

Run: python demo.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

from langchain_reasoner import LangChainReasoner
from kb_to_chromadb import ingest_kb_to_chromadb


def setup_kb():
    """Setup the knowledge base in ChromaDB if needed."""
    print("=" * 80)
    print("SETUP: Checking ChromaDB")
    print("=" * 80)
    
    db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # Check if ChromaDB already exists
    if os.path.exists(db_path) and os.listdir(db_path):
        print("✓ ChromaDB already exists")
        return
    
    print("ChromaDB not found. Ingesting knowledge base...")
    kb_path = "../task_4/regular_show_kb.pl"
    
    if not os.path.exists(kb_path):
        print(f"Error: KB file not found at {kb_path}")
        sys.exit(1)
    
    ingest_kb_to_chromadb(kb_path)
    print("✓ Knowledge base ingested successfully")


def main():
    """Run the demo with the Benson query."""
    
    # Setup KB
    setup_kb()
    
    print("\n" + "=" * 80)
    print("LANGCHAIN + RAG + BACKWARD CHAINING DEMO")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment")
        print("Please create a .env file in the project root with your API key")
        sys.exit(1)
    
    # Initialize reasoner
    print("\nInitializing LangChain Reasoner...")
    reasoner = LangChainReasoner()
    print("✓ Reasoner initialized")
    
    # The main query the user wants to test
    question = "Is Benson in charge of Mordecai?"
    
    print("\n" + "=" * 80)
    print("QUERY: " + question)
    print("=" * 80)
    
    # Run reasoning
    result = reasoner.reason(question, verbose=True)
    
    # Display final result
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"\nQuestion: {result['question']}")
    print(f"Query: {result['query']}")
    print(f"\n✓ RESULT: {result['result']}")
    
    if result['trace']:
        print("\nLogical Inference Trace:")
        print("-" * 80)
        print(result['trace'])
        print("-" * 80)
    
    print(f"\nKnowledge Base Statistics:")
    print(f"  - Facts used: {result['kb_stats']['facts']}")
    print(f"  - Rules used: {result['kb_stats']['rules']}")
    print(f"  - Total KB entries retrieved: {len(result['retrieved_kb'])}")
    
    # Additional test queries
    print("\n\n" + "=" * 80)
    print("ADDITIONAL TEST QUERIES")
    print("=" * 80)
    
    additional_questions = [
        "Are Mordecai and Rigby friends?",
        "Is Mordecai a park worker?",
        "Does Benson have authority?",
    ]
    
    for q in additional_questions:
        print(f"\n{'─' * 80}")
        print(f"Query: {q}")
        print('─' * 80)
        
        result = reasoner.reason(q, verbose=False)
        print(f"Result: {result['result']}")
        
        if result['trace']:
            print("\nTrace:")
            for line in result['trace'].split('\n'):
                if line.strip():
                    print(f"  {line}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nThe system successfully:")
    print("  ✓ Retrieved relevant facts/rules from ChromaDB using RAG")
    print("  ✓ Parsed natural language questions using LangChain")
    print("  ✓ Performed logical inference using backward chaining")
    print("  ✓ Generated TRUE/FALSE results with deduction traces")


if __name__ == "__main__":
    main()

