#!/usr/bin/env python3
"""
Demo: LangGraph + RAG + Backward Chaining with Relevancy Judgment and Self-Refinement

This demonstrates the complete LangGraph state machine pipeline:
1. RAG retrieval from ChromaDB
2. Relevancy judgment (LLM evaluates if docs are relevant)
3. Self-refinement (additional retrieval if needed)
4. NL parsing with LangChain
5. Backward chaining inference
6. Natural language answer generation

Key Improvements over Task 8:
- State machine architecture with LangGraph
- Automatic relevancy checking
- Self-refinement loop for better context
- More robust and observable execution

Run: python demo.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

# Import from task_8 for KB setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task_8'))
from kb_to_chromadb import ingest_kb_to_chromadb

from langgraph_reasoner import LangGraphReasoner


def setup_kb():
    """Setup the knowledge base in ChromaDB if needed."""
    print("=" * 80)
    print("SETUP: Checking ChromaDB")
    print("=" * 80)
    
    # Check if task_8 ChromaDB exists (we can reuse it)
    task8_db = os.path.join(os.path.dirname(__file__), '..', 'task_8', 'chroma_db')
    
    if os.path.exists(task8_db) and os.listdir(task8_db):
        print("✓ Using existing ChromaDB from task_8")
        # Update the KB path in query_kb to use task_8's DB
        return
    
    print("ChromaDB not found. Ingesting knowledge base...")
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'task_4', 'regular_show_kb.pl')
    
    if not os.path.exists(kb_path):
        print(f"Error: KB file not found at {kb_path}")
        sys.exit(1)
    
    # We'll use task_8's ChromaDB
    ingest_kb_to_chromadb(kb_path)
    print("✓ Knowledge base ingested successfully")


def main():
    """Run the LangGraph demo."""
    
    # Setup KB
    setup_kb()
    
    print("\n" + "=" * 80)
    print("LANGGRAPH REASONING SYSTEM DEMO")
    print("=" * 80)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment")
        print("Please create a .env file in the project root with your API key")
        sys.exit(1)
    
    # Initialize reasoner
    print("\nInitializing LangGraph Reasoner...")
    reasoner = LangGraphReasoner()
    print("✓ Reasoner initialized with state machine")
    
    # Demo queries
    demo_questions = [
        "Is Benson in charge of Mordecai?",
        "Are Mordecai and Rigby friends?",
        "Does Benson have authority?",
        "Who manages the park?",  # This might have low relevancy to test refinement
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING DEMO QUERIES")
    print("=" * 80)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'#' * 80}")
        print(f"QUERY {i}: {question}")
        print('#' * 80)
        
        result = reasoner.reason(question, verbose=True)
        
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"Question: {result['question']}")
        print(f"Result: {result['result']}")
        print(f"Relevancy Score: {result['relevancy_score']:.2f}")
        print(f"Refinement Iterations: {result['iterations']}")
        print(f"Documents Retrieved: {result['retrieved_count']}")
        print(f"\nNatural Language Answer:")
        print(f"  {result['final_answer']}")
        
        if i < len(demo_questions):
            input("\nPress Enter to continue to next query...")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nLangGraph State Machine Benefits:")
    print("  ✓ Automatic relevancy checking")
    print("  ✓ Self-refinement through additional retrieval")
    print("  ✓ Observable state transitions")
    print("  ✓ Conditional execution paths")
    print("  ✓ More robust error handling")
    print("\nComparison with Task 8:")
    print("  Task 8: Simple LangChain chains")
    print("  Task 9: LangGraph state machine with self-correction")


if __name__ == "__main__":
    main()

