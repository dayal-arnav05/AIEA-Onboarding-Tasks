# Task 9: LangGraph + RAG + Self-Refining Inference Engine

Advanced reasoning system using LangGraph state machines with automatic relevancy judgment and self-refinement capabilities.

## Overview

This task migrates Task 8's LangChain solution to **LangGraph**, adding:
- **Stateful execution** with observable state transitions
- **Relevancy judgment** using LLM to evaluate retrieved context
- **Self-refinement** through additional retrieval when relevancy is low
- **Conditional paths** based on relevancy scores
- **Chain of Thought** reasoning for complex queries

## Architecture

### LangGraph State Machine

```
START
  ↓
┌──────────────┐
│  Retrieve    │ ← RAG from ChromaDB
└──────┬───────┘
       ↓
┌──────────────────┐
│ Judge Relevancy  │ ← LLM scores 0.0-1.0
└──────┬───────────┘
       ↓
  [Decision Point]
       ├─ Score < 0.7? ──→ ┌─────────┐
       │                   │ Refine  │ ← Additional retrieval
       │                   └────┬────┘
       │                        ↓
       │                   [Re-judge]
       │                        ↓
       └─ Score ≥ 0.7 ──→ ┌─────────┐
                          │  Parse  │ ← NL → Prolog
                          └────┬────┘
                               ↓
                          ┌──────────┐
                          │ Build KB │
                          └────┬─────┘
                               ↓
                          ┌───────────┐
                          │  Infer    │ ← Backward chain
                          └────┬──────┘
                               ↓
                          ┌──────────┐
                          │  Format  │ ← NL answer
                          └────┬─────┘
                               ↓
                             END
```

## Key Improvements Over Task 8

### 1. State Machine vs Simple Chains

**Task 8 (LangChain):**
```python
# Linear execution
result = chain.invoke(input)
```

**Task 9 (LangGraph):**
```python
# Stateful with conditional branches
state = graph.invoke(initial_state)
# Can branch based on conditions
# Can loop back for refinement
```

### 2. Relevancy Judgment

Automatically evaluates if retrieved documents are sufficient:

```python
def judge_relevancy_node(state):
    """
    LLM judges: Are these docs relevant?
    - Score 0.0-1.0
    - Explanation why
    - Refinement suggestion if low
    """
    score, explanation, refinement = llm_judge(docs)
    
    if score < 0.7:
        # Trigger refinement path
        return {
            "needs_refinement": True,
            "refinement_query": refinement
        }
```

### 3. Self-Refinement Loop

If relevancy is low, automatically retrieves more context:

```python
def refine_node(state):
    """
    Get additional documents with refined query.
    Merge with existing docs.
    Re-judge relevancy.
    """
    more_docs = query_kb(state['refinement_query'])
    return {"retrieved_docs": combined_docs}
```

### 4. Observable State

Every node updates the shared state:

```python
class ReasoningState(TypedDict):
    question: str
    retrieved_docs: List[Dict]
    relevancy_score: float
    needs_refinement: bool
    parsed_fact: Fact
    inference_result: bool
    trace: str
    iteration: int
```

## Components

### State Definition

```python
class ReasoningState(TypedDict):
    """Complete state that flows through the graph."""
    question: str              # Original question
    retrieved_docs: List       # RAG results
    relevancy_score: float     # 0.0-1.0
    relevancy_explanation: str # Why this score?
    needs_refinement: bool     # Should we refine?
    refinement_query: str      # Better query
    parsed_fact: Fact          # Prolog query
    knowledge_base: KB         # Built KB
    inference_result: bool     # TRUE/FALSE
    trace: str                 # Deduction steps
    final_answer: str          # NL answer
    iteration: int             # Refinement count
```

### Graph Nodes

#### 1. retrieve_node
- Queries ChromaDB with natural language
- Returns top 15 relevant docs
- Uses refinement query if iterating

#### 2. judge_relevancy_node
- LLM evaluates document relevancy
- Scores 0.0 (irrelevant) to 1.0 (perfect)
- Checks for key entities, relationships, rules
- Suggests refinement if score < 0.7

#### 3. refine_node
- Triggered when relevancy is low
- Executes refined query
- Merges with existing docs
- Loops back to judge_relevancy

#### 4. parse_node
- Converts NL question → Prolog query
- Uses LLM with KB schema context
- Returns Fact object

#### 5. build_kb_node
- Constructs KnowledgeBase from retrieved docs
- Parses Prolog strings to Facts and Rules
- Prepares for inference

#### 6. infer_node
- Runs backward chaining
- Captures trace output
- Returns TRUE/FALSE result

#### 7. format_node
- Generates natural language answer
- Explains the reasoning
- Cites key deduction steps

### Conditional Edges

```python
def should_refine(state) -> str:
    """
    Decision function for conditional edge.
    
    Returns:
        "refine": If needs_refinement and iterations < max
        "parse": Otherwise, proceed to parsing
    """
    if state["needs_refinement"] and state["iteration"] < state["max_iterations"]:
        return "refine"
    return "parse"
```

## Usage

### Installation

```bash
cd task_9
pip install -r requirements.txt
```

### Run Demo

```bash
python demo.py
```

### Python API

```python
from langgraph_reasoner import LangGraphReasoner

# Initialize
reasoner = LangGraphReasoner()

# Reason with self-refinement
result = reasoner.reason("Is Benson in charge of Mordecai?")

# Access results
print(f"Result: {result['result']}")
print(f"Relevancy: {result['relevancy_score']}")
print(f"Iterations: {result['iterations']}")
print(f"Answer: {result['final_answer']}")
```

## Example Execution

### High Relevancy Query

```
Question: "Is Benson in charge of Mordecai?"

[Node: Retrieve]
  Retrieved 15 documents
  - Facts: 12, Rules: 3

[Node: Judge Relevancy]
  Relevancy Score: 0.85
  Explanation: Key entities present, in_charge_of rule found
  ✓ Relevancy sufficient

[Node: Parse]
  Parsed query: in_charge_of(benson, mordecai)

[Node: Build KB]
  - 12 facts, 3 rules

[Node: Infer]
  Goal: in_charge_of(benson, mordecai)
    Trying rule: in_charge_of(?Boss, ?Worker) :- reports_to(?Worker, ?Boss)
    Goal: reports_to(mordecai, benson)
      ✓ Matched fact
    ✓ Rule succeeded
  Result: TRUE

[Node: Format]
  Answer: "Yes, Benson is in charge of Mordecai because..."

RESULT: TRUE
Relevancy: 0.85
Iterations: 0 (no refinement needed)
```

### Low Relevancy Query (with Refinement)

```
Question: "Who manages the park?"

[Node: Retrieve]
  Retrieved 15 documents
  - Facts: 10, Rules: 2

[Node: Judge Relevancy]
  Relevancy Score: 0.55
  Explanation: Missing key predicates about management
  Low relevancy - refinement needed
  Refinement query: "park manager or boss authority"

[Node: Refine]
  Iteration 1
  Using refined query: "park manager or boss authority"
  Added 5 new documents
  Total documents: 20

[Node: Judge Relevancy]
  Relevancy Score: 0.90
  Explanation: Now includes park_manager facts and rules
  ✓ Relevancy sufficient

[Node: Parse]
  ...continues normally...

RESULT: TRUE
Relevancy: 0.90
Iterations: 1 (refined once)
```

## Key Features

### 1. Automatic Quality Control

The system self-evaluates and corrects:
- Low relevancy → Get more context
- Missing entities → Refine search
- Insufficient rules → Retrieve more

### 2. Observable Execution

Every node logs its actions:
```
[Node: Retrieve] → Retrieved 15 docs
[Node: Judge Relevancy] → Score: 0.85
[Node: Parse] → Query: in_charge_of(benson, mordecai)
[Node: Infer] → Result: TRUE
```

### 3. State Persistence

State flows through all nodes:
- Accumulates information
- Tracks iterations
- Maintains context

### 4. Conditional Branching

Different paths based on conditions:
- High relevancy → Direct to parsing
- Low relevancy → Refinement loop
- Max iterations → Force proceed

## Comparison: Task 8 vs Task 9

| Feature | Task 8 (LangChain) | Task 9 (LangGraph) |
|---------|-------------------|-------------------|
| **Architecture** | Linear chains | State machine |
| **Relevancy Check** | Manual | Automatic |
| **Self-Correction** | None | Automatic refinement |
| **State Management** | Implicit | Explicit TypedDict |
| **Branching** | None | Conditional edges |
| **Iteration** | None | Refinement loops |
| **Observability** | Limited | Full node-level |
| **Error Recovery** | Manual | Automatic retry |

## Technical Details

### State Flow

```python
initial_state = {
    "question": "Is Benson in charge?",
    "iteration": 0,
    "max_iterations": 3
}

# Node 1: retrieve
state["retrieved_docs"] = [...]

# Node 2: judge_relevancy
state["relevancy_score"] = 0.85
state["needs_refinement"] = False

# Decision: High relevancy → skip refine

# Node 3: parse
state["parsed_fact"] = Fact(...)

# ... continues
```

### Graph Compilation

```python
workflow = StateGraph(ReasoningState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("judge_relevancy", judge_relevancy_node)
# ... more nodes

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "judge_relevancy")

# Conditional edge
workflow.add_conditional_edges(
    "judge_relevancy",
    should_refine,  # Decision function
    {
        "refine": "refine",    # If needs refinement
        "parse": "parse"       # Otherwise
    }
)

graph = workflow.compile()
```

## Advanced Features

### Max Iterations

Prevents infinite refinement loops:
```python
reasoner.reason(question, max_iterations=3)
# Will refine at most 3 times
```

### Custom Relevancy Threshold

Can be adjusted for different use cases:
```python
# Current: score < 0.7 triggers refinement
# Could make configurable:
reasoner = LangGraphReasoner(relevancy_threshold=0.8)
```

### Chain of Thought

The judge_relevancy node uses CoT:
```
Consider:
1. Are key entities present?
2. Are relationships included?
3. Are inference rules available?

Score: 0.85
Reasoning: Found Benson (boss), Mordecai (worker), 
           reports_to relationship, and in_charge_of rule
```

## Benefits

1. **Reliability**: Self-corrects when context is insufficient
2. **Transparency**: Observable state at each step
3. **Flexibility**: Easy to add new nodes or modify paths
4. **Robustness**: Handles edge cases through refinement
5. **Explainability**: Clear reasoning trace with relevancy scores

## Limitations

- Requires more API calls for relevancy judgment
- Refinement loops add latency
- Max iterations prevents exhaustive search
- Relevancy scoring is subjective (LLM-based)

## Future Enhancements

- [ ] Cache relevancy scores for common queries
- [ ] Learn from refinement patterns
- [ ] Parallel node execution where possible
- [ ] User feedback loop for relevancy tuning
- [ ] Multi-agent collaboration nodes

## Dependencies

- **langgraph**: State machine framework
- **langchain**: LLM orchestration
- **chromadb**: Vector database
- **openai**: LLM API
- Task 7: Backward chaining engine
- Task 8: Prolog parser and ChromaDB setup

## Credits

This system demonstrates:
- **LangGraph**: Stateful AI workflows
- **Self-Refinement**: Automatic quality improvement
- **Neuro-Symbolic AI**: Neural + Symbolic reasoning
- **Explainable AI**: Transparent decision paths

