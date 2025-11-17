# Backward Chaining Inference Engine

A complete implementation of a backward chaining inference engine that works with Prolog-like rules to prove hypotheses from a knowledge base.

## Overview

Backward chaining is a goal-driven inference technique that works backward from a hypothesis to find supporting facts. This implementation includes:

- **Facts**: Ground truths in the knowledge base (e.g., `father(john, mary)`)
- **Rules**: Logical implications (e.g., `parent(X, Y) :- father(X, Y)`)
- **Variable Unification**: Pattern matching with variables (denoted by `?`)
- **Cycle Detection**: Prevents infinite loops in recursive rules
- **Trace Mode**: Shows the proof search process step-by-step

## Features

- ✅ Full backward chaining algorithm
- ✅ Variable unification and binding
- ✅ Multi-premise rules
- ✅ Cycle detection to handle recursive queries
- ✅ Depth limiting to prevent infinite recursion
- ✅ Variable renaming to avoid conflicts
- ✅ Trace mode for debugging
- ✅ Clean output for variable queries

## Usage

### Basic Example

```python
from backward_chain import KnowledgeBase, BackwardChainer, Fact, Rule

# Create a knowledge base
kb = KnowledgeBase()

# Add facts
kb.add_fact(Fact("father", "john", "mary"))
kb.add_fact(Fact("father", "john", "tom"))

# Add rules (conclusion :- premise1, premise2, ...)
kb.add_rule(Rule(
    conclusion=Fact("parent", "?X", "?Y"),
    premises=[Fact("father", "?X", "?Y")]
))

# Create a backward chainer
chainer = BackwardChainer(kb, trace=True)

# Prove a hypothesis
result = chainer.prove(Fact("parent", "john", "mary"))
print(f"Can prove: {result}")  # True

# Query with variables
bindings = chainer.prove_with_bindings(Fact("parent", "john", "?Y"))
print(f"John's children: {bindings}")
```

### Running the Demo

```bash
python backward_chain.py
```

This runs three demonstrations:
1. **Animal Classification**: Classifying animals as mammals, birds, and carnivores
2. **Family Relationships**: Inferring parent and grandparent relationships
3. **Variable Queries**: Finding all entities that match a pattern

## API Reference

### Classes

#### `Fact`
Represents a predicate with arguments.

```python
Fact(predicate: str, *args: str)
```

Example: `Fact("parent", "john", "mary")` or `Fact("parent", "?X", "?Y")`

#### `Rule`
Represents a logical rule (implication).

```python
Rule(conclusion: Fact, premises: List[Fact])
```

Example:
```python
Rule(
    conclusion=Fact("grandparent", "?X", "?Z"),
    premises=[
        Fact("parent", "?X", "?Y"),
        Fact("parent", "?Y", "?Z")
    ]
)
```

#### `KnowledgeBase`
Container for facts and rules.

**Methods:**
- `add_fact(fact: Fact)` - Add a fact to the knowledge base
- `add_rule(rule: Rule)` - Add a rule to the knowledge base

#### `BackwardChainer`
The backward chaining inference engine.

```python
BackwardChainer(kb: KnowledgeBase, trace: bool = False, max_depth: int = 50)
```

**Methods:**
- `prove(hypothesis: Fact) -> bool` - Check if a hypothesis can be proved
- `prove_with_bindings(hypothesis: Fact) -> List[Dict[str, str]]` - Prove and return all variable bindings
- `backchain_to_goal(hypothesis: Fact, bindings: Dict[str, str] = None) -> List[Dict[str, str]]` - Main backward chaining algorithm

**Parameters:**
- `trace`: Enable trace output to see the proof search process
- `max_depth`: Maximum recursion depth to prevent infinite loops

### Helper Functions

#### `extract_query_bindings`
Cleans up variable bindings by removing internal variables and resolving chains.

```python
extract_query_bindings(fact: Fact, bindings_list: List[Dict[str, str]]) -> List[Dict[str, str]]
```

## How It Works

### Backward Chaining Algorithm

1. **Goal**: Start with a goal (hypothesis) to prove
2. **Match Facts**: Check if the goal matches any known facts
3. **Try Rules**: For each rule whose conclusion matches the goal:
   - Unify the goal with the rule's conclusion
   - Recursively prove all premises of the rule
   - If all premises are proved, the goal is proved
4. **Cycle Detection**: Track goals being proved to avoid cycles
5. **Depth Limiting**: Stop if recursion depth exceeds limit

### Variable Unification

Variables (starting with `?`) are unified using pattern matching:
- `Fact("parent", "?X", "mary")` matches `Fact("parent", "john", "mary")` with binding `{?X: "john"}`
- Variables can bind to constants or other variables
- Variable renaming prevents conflicts between query and rule variables

### Example Trace

```
Goal: grandparent(john, alice)
  Trying rule: grandparent(?X, ?Z) :- parent(?X, ?Y), parent(?Y, ?Z)
  Goal: parent(john, ?Y_12)
    Trying rule: parent(?X, ?Y) :- father(?X, ?Y)
    Goal: father(john, ?Y_13)
      ✓ Matched fact: father(john, tom)
    ✓ Rule succeeded
  Goal: parent(tom, alice)
    Trying rule: parent(?X, ?Y) :- father(?X, ?Y)
    Goal: father(tom, alice)
      ✓ Matched fact: father(tom, alice)
    ✓ Rule succeeded
  ✓ Rule succeeded
Result: True
```

## Limitations

- **Recursive Rules**: Deep or unbounded recursion may hit the depth limit (default 50)
- **Combinatorial Explosion**: Variable queries with many solutions can be slow
- **No Negation**: Does not support negation-as-failure
- **No Built-ins**: No arithmetic or comparison operators

## Example Knowledge Bases

### Animal Classification

```python
kb = KnowledgeBase()
kb.add_fact(Fact("has_fur", "dog"))
kb.add_fact(Fact("warm_blooded", "dog"))
kb.add_rule(Rule(
    conclusion=Fact("mammal", "?X"),
    premises=[Fact("has_fur", "?X"), Fact("warm_blooded", "?X")]
))
```

### Family Relationships

```python
kb = KnowledgeBase()
kb.add_fact(Fact("father", "john", "mary"))
kb.add_fact(Fact("father", "john", "tom"))
kb.add_fact(Fact("father", "tom", "alice"))

kb.add_rule(Rule(
    conclusion=Fact("parent", "?X", "?Y"),
    premises=[Fact("father", "?X", "?Y")]
))

kb.add_rule(Rule(
    conclusion=Fact("grandparent", "?X", "?Z"),
    premises=[Fact("parent", "?X", "?Y"), Fact("parent", "?Y", "?Z")]
))
```

## Extending the System

To add new knowledge bases:

1. Create a new function that returns a `KnowledgeBase`
2. Add relevant facts using `kb.add_fact()`
3. Add rules using `kb.add_rule()`
4. Create a `BackwardChainer` and run queries

Example:
```python
def create_my_knowledge_base():
    kb = KnowledgeBase()
    # Add your facts and rules here
    return kb

kb = create_my_knowledge_base()
chainer = BackwardChainer(kb, trace=True)
result = chainer.prove(Fact("my_predicate", "arg1", "arg2"))
```

## Credits

This implementation demonstrates core AI concepts:
- Knowledge representation
- Logical inference
- Unification and pattern matching
- Search algorithms with cycle detection

