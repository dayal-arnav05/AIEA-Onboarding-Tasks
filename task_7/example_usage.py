"""
Example usage of the backward chaining system.
This shows how to create a custom knowledge base and run queries.
"""

from backward_chain import KnowledgeBase, BackwardChainer, Fact, Rule, extract_query_bindings


def create_simple_kb():
    """Create a simple knowledge base about programming languages."""
    kb = KnowledgeBase()
    
    # Facts about programming languages
    kb.add_fact(Fact("compiled", "c"))
    kb.add_fact(Fact("compiled", "rust"))
    kb.add_fact(Fact("compiled", "go"))
    kb.add_fact(Fact("interpreted", "python"))
    kb.add_fact(Fact("interpreted", "javascript"))
    kb.add_fact(Fact("interpreted", "ruby"))
    
    kb.add_fact(Fact("statically_typed", "c"))
    kb.add_fact(Fact("statically_typed", "rust"))
    kb.add_fact(Fact("statically_typed", "go"))
    kb.add_fact(Fact("dynamically_typed", "python"))
    kb.add_fact(Fact("dynamically_typed", "javascript"))
    kb.add_fact(Fact("dynamically_typed", "ruby"))
    
    kb.add_fact(Fact("has_gc", "go"))
    kb.add_fact(Fact("has_gc", "python"))
    kb.add_fact(Fact("has_gc", "javascript"))
    kb.add_fact(Fact("has_gc", "ruby"))
    
    # Rules
    kb.add_rule(Rule(
        conclusion=Fact("systems_language", "?X"),
        premises=[
            Fact("compiled", "?X"),
            Fact("statically_typed", "?X")
        ]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("scripting_language", "?X"),
        premises=[
            Fact("interpreted", "?X"),
            Fact("dynamically_typed", "?X")
        ]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("modern_systems_language", "?X"),
        premises=[
            Fact("systems_language", "?X"),
            Fact("has_gc", "?X")
        ]
    ))
    
    return kb


def main():
    print("=" * 70)
    print("CUSTOM KNOWLEDGE BASE EXAMPLE: Programming Languages")
    print("=" * 70)
    print()
    
    # Create knowledge base
    kb = create_simple_kb()
    print("Knowledge Base:")
    print(kb)
    
    # Create chainer with trace enabled
    chainer = BackwardChainer(kb, trace=True)
    
    # Example 1: Specific query
    print("\n--- Query 1: Is Rust a systems language? ---")
    result = chainer.prove(Fact("systems_language", "rust"))
    print(f"\nResult: {result}")
    
    # Example 2: Specific query
    print("\n\n--- Query 2: Is Python a scripting language? ---")
    result = chainer.prove(Fact("scripting_language", "python"))
    print(f"\nResult: {result}")
    
    # Example 3: Specific query
    print("\n\n--- Query 3: Is Go a modern systems language? ---")
    result = chainer.prove(Fact("modern_systems_language", "go"))
    print(f"\nResult: {result}")
    
    # Example 4: Variable query without trace
    print("\n\n" + "=" * 70)
    print("VARIABLE QUERIES")
    print("=" * 70)
    
    chainer_no_trace = BackwardChainer(kb, trace=False)
    
    print("\n--- Query 4: What are all the systems languages? ---")
    query = Fact("systems_language", "?X")
    bindings = chainer_no_trace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  {i}. {binding.get('?X', '?')}")
    
    print("\n--- Query 5: What are all the scripting languages? ---")
    query = Fact("scripting_language", "?X")
    bindings = chainer_no_trace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  {i}. {binding.get('?X', '?')}")
    
    print("\n--- Query 6: What languages have garbage collection? ---")
    query = Fact("has_gc", "?X")
    bindings = chainer_no_trace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  {i}. {binding.get('?X', '?')}")
    
    print("\n--- Query 7: What are all the modern systems languages? ---")
    query = Fact("modern_systems_language", "?X")
    bindings = chainer_no_trace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  {i}. {binding.get('?X', '?')}")


if __name__ == "__main__":
    main()

