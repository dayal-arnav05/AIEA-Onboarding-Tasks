#!/usr/bin/env python3
"""
Quick test script for Logic-LM to verify setup
"""

import os
import sys
from logic_lm import LogicLM


def test_logic_lm():
    """Test Logic-LM with a simple question."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("Make sure it's set in the root .env file")
        return False
    
    # Check for knowledge base
    kb_path = "../task_4/regular_show_kb.pl"
    if not os.path.exists(kb_path):
        print(f"❌ Error: Knowledge base not found at {kb_path}")
        return False
    
    print("🔧 Testing Logic-LM setup...\n")
    
    try:
        # Initialize Logic-LM
        print("Initializing Logic-LM...")
        logic_lm = LogicLM(kb_path=kb_path, model="gpt-4")
        print("✓ Logic-LM initialized\n")
        
        # Test a simple question
        test_question = "Who are the park workers?"
        print(f"Testing with question: '{test_question}'\n")
        
        result = logic_lm.reason(test_question, verbose=True)
        
        if result["success"]:
            print("\n✅ Test passed! Logic-LM is working correctly.")
            return True
        else:
            print("\n❌ Test failed: Query did not succeed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_logic_lm()
    sys.exit(0 if success else 1)

