#!/usr/bin/env python3
"""
Quick demo of Logic-LM capabilities
Shows various types of queries and reasoning
"""

from logic_lm import LogicLM
import os
import sys

def main():
    """Run a comprehensive demo of Logic-LM."""
    
    kb_path = "../task_4/regular_show_kb.pl"

    
    # Initialize
    print("Logic-LM Demo: Neurosymbolic Reasoning on Regular Show Knowledge Base")
    print("Paper: https://arxiv.org/abs/2305.12295")
  
    
    logic_lm = LogicLM(kb_path=kb_path, model="gpt-4")
    
    # Demo queries organized by type
    demos = [
        {
            "category": "Basic Fact Queries",
            "questions": [
                "Who are all the park workers?",
                "Who is the boss?",
            ]
        },
        {
            "category": "Relationship Queries",
            "questions": [
                "Are Mordecai and Rigby friends?",
                "Who does Rigby report to?",
            ]
        },
        {
            "category": "Rule-Based Reasoning",
            "questions": [
                "Who does Benson manage?",
                "Who has authority in the park?",
            ]
        },
        {
            "category": "Character Information",
            "questions": [
                "What type of character is Mordecai?",
                "Is Benson a park worker?",
            ]
        }
    ]
    
    # Run demos
    for demo in demos:
        print("\n" + "="*80)
        print(demo["category"])
        print("="*80)
        
        for question in demo["questions"]:
            result = logic_lm.reason(question, verbose=True)
            
            # Show reasoning summary
            if result["success"]:
                print(f"\n  Reasoning Summary:")
                print(f"     Attempts: {len(result['attempts'])}")
                print(f"     Final Query: {result['attempts'][-1]['prolog_query']}")
                print(f"     Result Count: {len(result['attempts'][-1]['results'])}")
    
    # Final summary
    print("Demo Complete!")
    print(f"\nTotal questions processed: {len(logic_lm.history)}")
    successful = sum(1 for r in logic_lm.history if r['success'])
    print(f"Successful: {successful}/{len(logic_lm.history)}")
    
    # Show self-refinement stats
    refinements = sum(1 for r in logic_lm.history if len(r['attempts']) > 1)
    if refinements > 0:
        print(f"Questions requiring refinement: {refinements}")



if __name__ == "__main__":
    main()

