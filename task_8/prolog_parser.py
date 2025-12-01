"""
Prolog Parser: Converts Prolog syntax strings to Python Fact/Rule objects.

This module provides functions to parse Prolog knowledge base files (.pl) and convert
them into Python objects (Fact and Rule) that can be used by the backward chaining
inference engine.

Key Features:
- Parse individual Prolog facts: "park_worker(mordecai)" → Fact("park_worker", "mordecai")
- Parse Prolog rules: "in_charge_of(X,Y) :- reports_to(Y,X)" → Rule(...)
- Convert Prolog variables (uppercase) to Python format (?-prefixed)
- Handle multi-line Prolog statements
- Parse entire .pl files into facts and rules lists

Critical for:
- Loading knowledge bases from Prolog files
- Converting retrieved ChromaDB documents back to objects
- Bridging Prolog syntax with Python inference engine
"""

import re
import sys
import os
from typing import List, Tuple, Optional

# Import from task_7
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task_7'))
from backward_chain import Fact, Rule


def parse_prolog_fact(fact_str: str) -> Optional[Fact]:
    """
    Parse a Prolog fact string into a Fact object.
    
    Examples:
        "park_worker(mordecai)" -> Fact("park_worker", "mordecai")
        "friends(mordecai, rigby)" -> Fact("friends", "mordecai", "rigby")
        "parent(X, Y)" -> Fact("parent", "?X", "?Y")  # Convert uppercase vars to ?-prefixed
    
    Args:
        fact_str: Prolog fact string
        
    Returns:
        Fact object or None if parsing fails
    """
    fact_str = fact_str.strip().rstrip('.')
    
    # Match predicate(arg1, arg2, ...)
    match = re.match(r'(\w+)\((.*?)\)', fact_str)
    if not match:
        return None
    
    predicate = match.group(1)
    args_str = match.group(2)
    
    if not args_str:
        return Fact(predicate)
    
    # Split arguments by comma (handle nested parentheses if needed)
    args_raw = [arg.strip() for arg in args_str.split(',')]
    
    # Convert Prolog variables (uppercase or _) to Python format (?-prefixed)
    args = []
    for arg in args_raw:
        if arg == '_':
            # Anonymous variable
            args.append('?_')
        elif arg and arg[0].isupper():
            # Prolog variable -> Python variable
            args.append(f'?{arg}')
        else:
            # Constant (atom)
            args.append(arg)
    
    return Fact(predicate, *tuple(args))


def split_premises(premises_str: str) -> List[str]:
    """
    Split premises by comma, respecting parentheses.
    
    Args:
        premises_str: String of premises separated by commas
        
    Returns:
        List of individual premise strings
    """
    premises = []
    current = ""
    paren_depth = 0
    
    for char in premises_str:
        if char == '(':
            paren_depth += 1
            current += char
        elif char == ')':
            paren_depth -= 1
            current += char
        elif char == ',' and paren_depth == 0:
            # Split here
            premises.append(current.strip())
            current = ""
        else:
            current += char
    
    # Add the last premise
    if current.strip():
        premises.append(current.strip())
    
    return premises


def parse_prolog_rule(rule_str: str) -> Optional[Rule]:
    """
    Parse a Prolog rule string into a Rule object.
    
    Examples:
        "in_charge_of(Boss, Worker) :- reports_to(Worker, Boss)"
        -> Rule(conclusion=Fact(...), premises=[Fact(...)])
    
    Args:
        rule_str: Prolog rule string
        
    Returns:
        Rule object or None if parsing fails
    """
    rule_str = rule_str.strip().rstrip('.')
    
    # Split by :-
    if ':-' not in rule_str:
        return None
    
    parts = rule_str.split(':-')
    if len(parts) != 2:
        return None
    
    conclusion_str = parts[0].strip()
    premises_str = parts[1].strip()
    
    # Parse conclusion
    conclusion = parse_prolog_fact(conclusion_str)
    if not conclusion:
        return None
    
    # Parse premises (split by comma, respecting parentheses)
    premise_strs = split_premises(premises_str)
    premises = []
    
    for premise_str in premise_strs:
        # Skip special predicates like \= (not equal)
        if '\\=' in premise_str or premise_str.startswith('_'):
            continue
            
        premise = parse_prolog_fact(premise_str)
        if premise:
            premises.append(premise)
    
    return Rule(conclusion=conclusion, premises=premises)


def parse_prolog_line(line: str) -> Optional[Tuple[str, object]]:
    """
    Parse a single line from a Prolog file.
    
    Returns:
        ("fact", Fact) or ("rule", Rule) or None
    """
    line = line.strip()
    
    # Skip comments and empty lines
    if not line or line.startswith('%'):
        return None
    
    # Check if it's a rule (contains :-)
    if ':-' in line:
        rule = parse_prolog_rule(line)
        if rule:
            return ("rule", rule)
    else:
        # It's a fact
        fact = parse_prolog_fact(line)
        if fact:
            return ("fact", fact)
    
    return None


def parse_prolog_file(filepath: str) -> Tuple[List[Fact], List[Rule]]:
    """
    Parse an entire Prolog file into facts and rules.
    
    Args:
        filepath: Path to Prolog file
        
    Returns:
        (facts, rules) tuple
    """
    facts = []
    rules = []
    
    with open(filepath, 'r') as f:
        # Handle multi-line rules
        current_statement = ""
        
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('%'):
                continue
            
            # Accumulate multi-line statements
            current_statement += " " + line
            
            # If statement ends with period, process it
            if current_statement.strip().endswith('.'):
                result = parse_prolog_line(current_statement)
                if result:
                    kind, obj = result
                    if kind == "fact":
                        facts.append(obj)
                    elif kind == "rule":
                        rules.append(obj)
                
                current_statement = ""
    
    return facts, rules


def convert_to_python_variable(prolog_var: str) -> str:
    """
    Convert Prolog variable (uppercase) to Python variable format (?X).
    
    Examples:
        "X" -> "?X"
        "Boss" -> "?Boss"
        "mordecai" -> "mordecai" (not a variable)
    """
    if prolog_var and prolog_var[0].isupper():
        return f"?{prolog_var}"
    return prolog_var


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Parse file provided as argument
        kb_path = sys.argv[1]
        if os.path.exists(kb_path):
            print(f"Parsing {kb_path}...")
            facts, rules = parse_prolog_file(kb_path)
            print(f"✓ Parsed {len(facts)} facts and {len(rules)} rules")
        else:
            print(f"Error: File not found: {kb_path}")
    else:
        print("Usage: python prolog_parser.py <path_to_prolog_file>")
        print("Example: python prolog_parser.py ../task_4/regular_show_kb.pl")

