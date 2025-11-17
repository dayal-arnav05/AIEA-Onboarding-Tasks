"""
Backward Chaining Implementation

This module implements a backward chaining inference engine that works
with Prolog-like rules to prove hypotheses from a knowledge base.
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Fact:
    """Represents a fact with a predicate and arguments."""
    predicate: str
    args: Tuple[str, ...]
    
    def __init__(self, predicate: str, *args: str):
        self.predicate = predicate
        self.args = args
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def __eq__(self, other):
        if not isinstance(other, Fact):
            return False
        return self.predicate == other.predicate and self.args == other.args
    
    def __repr__(self):
        if self.args:
            return f"{self.predicate}({', '.join(self.args)})"
        return self.predicate
    
    def substitute(self, bindings: Dict[str, str]) -> 'Fact':
        """Apply variable bindings to this fact, following chains of bindings."""
        new_args = []
        for arg in self.args:
            # Follow the chain of bindings to get the final value
            value = arg
            seen = {arg}
            while value in bindings:
                if value in seen and value != arg:
                    break  # Avoid infinite loops
                next_value = bindings[value]
                if next_value == value:
                    break
                seen.add(value)
                value = next_value
            new_args.append(value)
        return Fact(self.predicate, *tuple(new_args))
    
    def _resolve_binding(self, var: str, bindings: Dict[str, str]) -> str:
        """Follow the chain of bindings to get the final value."""
        value = var
        seen = {var}
        while value in bindings:
            next_value = bindings[value]
            if next_value == value or next_value in seen:
                break
            seen.add(next_value)
            value = next_value
        return value
    
    def match(self, other: 'Fact', bindings: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
        """
        Try to match this fact with another, returning variable bindings if successful.
        Variables start with '?'
        """
        if bindings is None:
            bindings = {}
        else:
            bindings = bindings.copy()
        
        if self.predicate != other.predicate:
            return None
        
        if len(self.args) != len(other.args):
            return None
        
        for self_arg, other_arg in zip(self.args, other.args):
            # Resolve bindings for both arguments
            resolved_self = self._resolve_binding(self_arg, bindings) if self_arg.startswith('?') else self_arg
            resolved_other = self._resolve_binding(other_arg, bindings) if other_arg.startswith('?') else other_arg
            
            # If both are variables (after resolution)
            if resolved_self.startswith('?') and resolved_other.startswith('?'):
                if resolved_self != resolved_other:
                    # Bind one to the other
                    bindings[resolved_self] = resolved_other
            # If resolved_self is a variable
            elif resolved_self.startswith('?'):
                bindings[resolved_self] = resolved_other
            # If resolved_other is a variable
            elif resolved_other.startswith('?'):
                bindings[resolved_other] = resolved_self
            # Both are constants
            else:
                if resolved_self != resolved_other:
                    return None
        
        return bindings


@dataclass
class Rule:
    """Represents a rule: conclusion :- premise1, premise2, ..., premiseN"""
    conclusion: Fact
    premises: List[Fact]
    
    def __repr__(self):
        if not self.premises:
            return f"{self.conclusion}"
        premises_str = ", ".join(str(p) for p in self.premises)
        return f"{self.conclusion} :- {premises_str}"


class KnowledgeBase:
    """A knowledge base containing facts and rules."""
    
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
    
    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base."""
        self.rules.append(rule)
    
    def __repr__(self):
        result = "Facts:\n"
        for fact in self.facts:
            result += f"  {fact}\n"
        result += "\nRules:\n"
        for rule in self.rules:
            result += f"  {rule}\n"
        return result


class BackwardChainer:
    """Implements backward chaining inference."""
    
    def __init__(self, kb: KnowledgeBase, trace: bool = False, max_depth: int = 50):
        self.kb = kb
        self.trace = trace
        self.trace_depth = 0
        self.proved_goals: Set[Fact] = set()
        self.rule_counter = 0
        self.max_depth = max_depth  # Maximum recursion depth
        self.current_goals: List[Fact] = []  # Stack of goals currently being proved (for cycle detection)
    
    def _trace_print(self, message: str):
        """Print a trace message with proper indentation."""
        if self.trace:
            print("  " * self.trace_depth + message)
    
    def _rename_rule_variables(self, rule: Rule) -> Rule:
        """
        Rename all variables in a rule to avoid collisions with query variables.
        Creates a fresh copy of the rule with renamed variables.
        """
        self.rule_counter += 1
        suffix = f"_{self.rule_counter}"
        
        # Find all variables in the rule
        variables = set()
        variables.update(arg for arg in rule.conclusion.args if arg.startswith('?'))
        for premise in rule.premises:
            variables.update(arg for arg in premise.args if arg.startswith('?'))
        
        # Create renaming map
        renaming = {var: var + suffix for var in variables}
        
        # Apply renaming
        new_conclusion = Fact(
            rule.conclusion.predicate,
            *[renaming.get(arg, arg) for arg in rule.conclusion.args]
        )
        new_premises = [
            Fact(premise.predicate, *[renaming.get(arg, arg) for arg in premise.args])
            for premise in rule.premises
        ]
        
        return Rule(new_conclusion, new_premises)
    
    def backchain_to_goal(self, hypothesis: Fact, bindings: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Backward chain to prove a hypothesis.
        
        Args:
            hypothesis: The goal/hypothesis to prove
            bindings: Current variable bindings
            
        Returns:
            List of binding dictionaries that prove the hypothesis (empty list if unprovable)
        """
        if bindings is None:
            bindings = {}
        
        self._trace_print(f"Goal: {hypothesis.substitute(bindings)}")
        self.trace_depth += 1
        
        results = []
        
        # Apply current bindings to hypothesis
        bound_hypothesis = hypothesis.substitute(bindings)
        
        # Check depth limit to prevent infinite recursion
        if self.trace_depth > self.max_depth:
            self._trace_print(f"✗ Max depth reached: {bound_hypothesis}")
            self.trace_depth -= 1
            return []
        
        # Check for cycles - if we're already trying to prove this goal
        if bound_hypothesis in self.current_goals:
            self._trace_print(f"✗ Cycle detected: {bound_hypothesis}")
            self.trace_depth -= 1
            return []
        
        # Check if this goal has already been proved
        if bound_hypothesis in self.proved_goals:
            self._trace_print(f"Already proved: {bound_hypothesis}")
            self.trace_depth -= 1
            return [bindings]
        
        # Add to current goals (cycle detection)
        self.current_goals.append(bound_hypothesis)
        
        # Step 1: Check if the hypothesis matches any known facts
        for fact in self.kb.facts:
            # Try to unify the hypothesis with the fact
            match_bindings = hypothesis.match(fact, bindings)
            if match_bindings is not None:
                self._trace_print(f"✓ Matched fact: {fact}")
                matched_hypothesis = hypothesis.substitute(match_bindings)
                self.proved_goals.add(matched_hypothesis)
                results.append(match_bindings)
            elif self.trace and fact.predicate == hypothesis.predicate:
                # Debug: show why match failed for same predicate
                pass  # Silently skip to avoid clutter
        
        # Step 2: Try to prove using rules (backward chain through rules)
        for rule in self.kb.rules:
            # Rename rule variables to avoid collisions with query variables
            renamed_rule = self._rename_rule_variables(rule)
            
            # Try to match the hypothesis with the rule's conclusion
            match_bindings = hypothesis.match(renamed_rule.conclusion, bindings)
            if match_bindings is not None:
                self._trace_print(f"Trying rule: {rule}")
                
                # Try to prove all premises of the rule
                premise_results = self._prove_premises(renamed_rule.premises, match_bindings)
                
                if premise_results:
                    self._trace_print(f"✓ Rule succeeded: {rule}")
                    matched_hypothesis = hypothesis.substitute(match_bindings)
                    self.proved_goals.add(matched_hypothesis)
                    results.extend(premise_results)
                else:
                    self._trace_print(f"✗ Rule failed: {rule}")
        
        if not results:
            self._trace_print(f"✗ Cannot prove: {bound_hypothesis}")
        
        # Remove from current goals
        self.current_goals.pop()
        
        self.trace_depth -= 1
        return results
    
    def _prove_premises(self, premises: List[Fact], bindings: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Try to prove all premises of a rule.
        
        Args:
            premises: List of premises to prove
            bindings: Current variable bindings
            
        Returns:
            List of binding dictionaries that prove all premises (empty if any premise fails)
        """
        if not premises:
            return [bindings]
        
        # Try to prove the first premise
        first_premise = premises[0]
        remaining_premises = premises[1:]
        
        first_results = self.backchain_to_goal(first_premise, bindings)
        
        if not first_results:
            return []
        
        # For each solution to the first premise, try to prove the remaining premises
        all_results = []
        for result_bindings in first_results:
            if remaining_premises:
                remaining_results = self._prove_premises(remaining_premises, result_bindings)
                all_results.extend(remaining_results)
            else:
                all_results.append(result_bindings)
        
        return all_results
    
    def prove(self, hypothesis: Fact) -> bool:
        """
        Simple yes/no check if a hypothesis can be proved.
        
        Args:
            hypothesis: The hypothesis to prove
            
        Returns:
            True if provable, False otherwise
        """
        self.proved_goals.clear()
        results = self.backchain_to_goal(hypothesis)
        return len(results) > 0
    
    def prove_with_bindings(self, hypothesis: Fact) -> List[Dict[str, str]]:
        """
        Prove a hypothesis and return all possible variable bindings.
        
        Args:
            hypothesis: The hypothesis to prove
            
        Returns:
            List of binding dictionaries
        """
        self.proved_goals.clear()
        return self.backchain_to_goal(hypothesis)


def create_animal_knowledge_base() -> KnowledgeBase:
    """
    Create a sample knowledge base about animals.
    
    Rules:
    - mammal(X) :- has_fur(X), warm_blooded(X)
    - bird(X) :- has_feathers(X), warm_blooded(X)
    - can_fly(X) :- bird(X), not flightless(X)
    - carnivore(X) :- eats(X, meat)
    """
    kb = KnowledgeBase()
    
    # Facts
    kb.add_fact(Fact("has_fur", "dog"))
    kb.add_fact(Fact("warm_blooded", "dog"))
    kb.add_fact(Fact("eats", "dog", "meat"))
    
    kb.add_fact(Fact("has_feathers", "sparrow"))
    kb.add_fact(Fact("warm_blooded", "sparrow"))
    
    kb.add_fact(Fact("has_feathers", "penguin"))
    kb.add_fact(Fact("warm_blooded", "penguin"))
    kb.add_fact(Fact("flightless", "penguin"))
    
    kb.add_fact(Fact("has_fur", "cat"))
    kb.add_fact(Fact("warm_blooded", "cat"))
    kb.add_fact(Fact("eats", "cat", "meat"))
    
    # Rules
    kb.add_rule(Rule(
        conclusion=Fact("mammal", "?X"),
        premises=[
            Fact("has_fur", "?X"),
            Fact("warm_blooded", "?X")
        ]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("bird", "?X"),
        premises=[
            Fact("has_feathers", "?X"),
            Fact("warm_blooded", "?X")
        ]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("carnivore", "?X"),
        premises=[
            Fact("eats", "?X", "meat")
        ]
    ))
    
    return kb


def create_family_knowledge_base() -> KnowledgeBase:
    """
    Create a sample knowledge base about family relationships.
    
    Rules:
    - parent(X, Y) :- father(X, Y)
    - parent(X, Y) :- mother(X, Y)
    - grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    """
    kb = KnowledgeBase()
    
    # Facts
    kb.add_fact(Fact("father", "john", "mary"))
    kb.add_fact(Fact("father", "john", "tom"))
    kb.add_fact(Fact("mother", "susan", "mary"))
    kb.add_fact(Fact("mother", "susan", "tom"))
    kb.add_fact(Fact("father", "tom", "alice"))
    kb.add_fact(Fact("mother", "jane", "alice"))
    
    # Rules
    kb.add_rule(Rule(
        conclusion=Fact("parent", "?X", "?Y"),
        premises=[Fact("father", "?X", "?Y")]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("parent", "?X", "?Y"),
        premises=[Fact("mother", "?X", "?Y")]
    ))
    
    kb.add_rule(Rule(
        conclusion=Fact("grandparent", "?X", "?Z"),
        premises=[
            Fact("parent", "?X", "?Y"),
            Fact("parent", "?Y", "?Z")
        ]
    ))
    
    return kb


def extract_query_bindings(fact: Fact, bindings_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract only the bindings for variables in the original query,
    filtering out internal renamed variables and resolving chains.
    """
    # Find variables in the original query
    query_vars = set(arg for arg in fact.args if arg.startswith('?'))
    
    result = []
    for bindings in bindings_list:
        # Resolve binding chains and filter to query variables only
        clean_bindings = {}
        for var in query_vars:
            if var in bindings:
                # Follow the chain of bindings to get the final value
                value = bindings[var]
                seen = {var}
                while value in bindings and value.startswith('?'):
                    if value in seen:
                        break  # Avoid cycles
                    seen.add(value)
                    value = bindings[value]
                # Only keep if we found a concrete value (not a variable)
                if not value.startswith('?'):
                    clean_bindings[var] = value
        
        # Only include if we found actual bindings
        if clean_bindings:
            result.append(clean_bindings)
    
    # Remove duplicates
    unique_results = []
    seen_bindings = set()
    for binding in result:
        binding_tuple = tuple(sorted(binding.items()))
        if binding_tuple not in seen_bindings:
            seen_bindings.add(binding_tuple)
            unique_results.append(binding)
    
    return unique_results


def demo():
    """Demonstrate the backward chaining system."""
    
    print("=" * 70)
    print("BACKWARD CHAINING DEMONSTRATION")
    print("=" * 70)
    
    # Demo 1: Animal Knowledge Base
    print("\n### DEMO 1: Animal Classification ###\n")
    kb1 = create_animal_knowledge_base()
    print(kb1)
    
    chainer1 = BackwardChainer(kb1, trace=True)
    
    print("\n--- Query 1: Is a dog a mammal? ---")
    result = chainer1.prove(Fact("mammal", "dog"))
    print(f"\nResult: {result}\n")
    
    print("\n--- Query 2: Is a sparrow a bird? ---")
    result = chainer1.prove(Fact("bird", "sparrow"))
    print(f"\nResult: {result}\n")
    
    print("\n--- Query 3: Is a cat a carnivore? ---")
    result = chainer1.prove(Fact("carnivore", "cat"))
    print(f"\nResult: {result}\n")
    
    print("\n--- Query 4: Is a penguin a bird? ---")
    result = chainer1.prove(Fact("bird", "penguin"))
    print(f"\nResult: {result}\n")
    
    # Demo 2: Family Relationships
    print("\n" + "=" * 70)
    print("### DEMO 2: Family Relationships ###\n")
    kb2 = create_family_knowledge_base()
    print(kb2)
    
    chainer2 = BackwardChainer(kb2, trace=True)
    
    print("\n--- Query 1: Is John a parent of Mary? ---")
    result = chainer2.prove(Fact("parent", "john", "mary"))
    print(f"\nResult: {result}\n")
    
    print("\n--- Query 2: Is John a grandparent of Alice? ---")
    result = chainer2.prove(Fact("grandparent", "john", "alice"))
    print(f"\nResult: {result}\n")
    
    print("\n--- Query 3: Is Susan a parent of Tom? ---")
    result = chainer2.prove(Fact("parent", "susan", "tom"))
    print(f"\nResult: {result}\n")
    
    # Demo 3: Variable Queries
    print("\n" + "=" * 70)
    print("### DEMO 3: Variable Queries ###\n")
    
    print("--- Query: Who are the parents of Alice? ---")
    chainer2_noTrace = BackwardChainer(kb2, trace=False)
    query = Fact("parent", "?X", "alice")
    bindings = chainer2_noTrace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  Solution {i}: ?X = {binding.get('?X', '?')}")
    
    print("\n--- Query: Who are John's children (via parent relation)? ---")
    query = Fact("parent", "john", "?Y")
    bindings = chainer2_noTrace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  Solution {i}: ?Y = {binding.get('?Y', '?')}")
    
    print("\n--- Query: Who are the grandparents of Alice? ---")
    query = Fact("grandparent", "?X", "alice")
    bindings = chainer2_noTrace.prove_with_bindings(query)
    clean_bindings = extract_query_bindings(query, bindings)
    print(f"Found {len(clean_bindings)} solution(s):")
    for i, binding in enumerate(clean_bindings, 1):
        print(f"  Solution {i}: ?X = {binding.get('?X', '?')}")


if __name__ == "__main__":
    demo()

