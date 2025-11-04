#!/usr/bin/env python3
"""
Regular Show Knowledge Base Interpreter using SWI-Prolog
This script demonstrates parsing and querying a Prolog knowledge base
about Regular Show characters and their relationships.
"""

import os
from pyswip import Prolog

def main():
    # Initialize Prolog
    prolog = Prolog()
    
    # Load the knowledge base using consult function from pyswip
    kb_path = os.path.abspath("regular_show_kb.pl")
    prolog.consult(kb_path)
    
    # Sample Query 1: List all park workers using the park_worker predicate
    workers = list(prolog.query("park_worker(X)"))
    for worker in workers:
        print(f"  • {worker['X'].capitalize()}")
    print()
    
    # Sample Query 2: Who is the boss?
    print("Query 2: Who is the boss?")
    boss_result = list(prolog.query("boss(X)"))
    for boss in boss_result:
        print(f"  • {boss['X'].capitalize()}")
    print()
    
    # Sample Query 3: Who does Mordecai report to?
    print("Query 3: Who does Mordecai report to?")
    reports = list(prolog.query("reports_to(mordecai, X)"))
    for report in reports:
        print(f"  • Mordecai reports to {report['X'].capitalize()}")
    print()
    
    # Sample Query 4: Who is Benson in charge of? (using the rule)
    print("Query 4: Who is Benson in charge of? (using in_charge_of rule)")
    in_charge = list(prolog.query("in_charge_of(benson, X)"))
    for person in in_charge:
        print(f"  • {person['X'].capitalize()}")
    print()
    
    # Sample Query 5: Are Mordecai and Rigby friends?
    print("Query 5: Are Mordecai and Rigby friends?")
    friendship = list(prolog.query("friends(mordecai, rigby)"))
    if friendship:
        print("  • Yes, they are friends!")
    else:
        print("  • No, they are not friends.")
    print()
    
    # Sample Query 6: What type of character is each person?
    print("Query 6: What type of character is everyone?")
    char_types = list(prolog.query("character_type(X, Y)"))
    for char in char_types:
        print(f"  • {char['X'].capitalize()} is a {char['Y'].replace('_', ' ')}")
    print()
    
    # Sample Query 7: Who has authority?
    print("Query 7: Who has authority? (using has_authority rule)")
    authorities = list(prolog.query("has_authority(X)"))
    for auth in authorities:
        print(f"  • {auth['X'].capitalize()}")
    print()
    
    # Sample Query 8: Who are subordinates?
    print("Query 8: Who are subordinates? (using is_subordinate rule)")
    subordinates = list(prolog.query("is_subordinate(X)"))
    for sub in subordinates:
        print(f"  • {sub['X'].capitalize()}")
    print()
    
    # Sample Query 9: Who works together with Mordecai?
    print("Query 9: Who works together with Mordecai? (using work_together rule)")
    coworkers = list(prolog.query("work_together(mordecai, X)"))
    for coworker in coworkers:
        print(f"  • {coworker['X'].capitalize()}")
    print()
    
    # Sample Query 10: Who are Muscle Man's friends?
    print("Query 10: Who are Muscle Man's friends?")
    mm_friends = list(prolog.query("friends(muscle_man, X)"))
    for friend in mm_friends:
        print(f"  • {friend['X'].replace('_', ' ').title()}")
    print()
    
    #Complex query - Find all boss-worker pairs
    print("Complex Query: Show all boss-worker relationships")
    relationships = list(prolog.query("in_charge_of(Boss, Worker)"))
    for rel in relationships:
        boss_name = rel['Boss'].capitalize()
        worker_name = rel['Worker'].capitalize()
        print(f"  • {boss_name} is in charge of {worker_name}")
    
if __name__ == "__main__":
    main()

