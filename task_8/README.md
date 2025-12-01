# Task 8: LangChain + RAG + Backward Chaining Inference Engine

A complete reasoning system that integrates:
- **LangChain**: For natural language processing and orchestration
- **RAG (ChromaDB)**: For knowledge base retrieval
- **Backward Chaining**: For logical inference with traces

## Overview

This system takes natural language questions about the Regular Show knowledge base and produces **TRUE/FALSE** answers with **logical inference traces** showing the deduction steps.

### Architecture

```
Natural Language Question
         ↓
┌─────────────────────────────────────┐
│  1. RAG: Query ChromaDB             │
│     Retrieve relevant facts/rules   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  2. LangChain: Parse Question       │
│     Convert NL → Fact() object      │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  3. Build Dynamic KB                │
│     From retrieved documents        │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  4. Backward Chaining Inference     │
│     Prove with trace enabled        │
└─────────────────┬───────────────────┘
                  ↓
    TRUE/FALSE + Deduction Trace
```

## Components

### 1. `prolog_parser.py`
Parses Prolog syntax into Python `Fact` and `Rule` objects from the backward chainer.

**Features:**
- Parse Prolog facts: `friends(mordecai, rigby)` → `Fact("friends", "mordecai", "rigby")`
- Parse Prolog rules: `in_charge_of(X,Y) :- reports_to(Y,X)` → `Rule(...)`
- Parse entire Prolog files

### 2. `kb_to_chromadb.py`
Ingests the Regular Show knowledge base into ChromaDB for vector retrieval.

**Features:**
- Converts Prolog facts/rules to natural language descriptions
- Stores with metadata (type, predicate, original Prolog)
- Supports semantic search over KB

### 3. `langchain_reasoner.py`
Main reasoning system using LangChain.

**Pipeline:**
1. **RAG Retrieval**: Query ChromaDB for relevant KB entries
2. **LangChain Parsing**: Convert NL question → `Fact()` object
3. **KB Building**: Construct `KnowledgeBase` from retrieved documents
4. **Backward Chaining**: Run inference with trace
5. **Result Formatting**: Return TRUE/FALSE with trace

**LangChain Components:**
- `ChatOpenAI`: LLM for parsing and formatting
- `ChatPromptTemplate`: Structured prompts
- `RunnableSequence`: Processing pipeline

### 4. `demo.py`
Demonstration script showing the complete system.

## Setup

### 1. Install Dependencies

```bash
cd task_8
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Run Demo

```bash
python demo.py
```

## Usage

### Basic Query

```python
from langchain_reasoner import LangChainReasoner

reasoner = LangChainReasoner()
result = reasoner.reason("Is Benson in charge of Mordecai?")

print(f"Result: {result['result']}")  # True
print(f"Trace: {result['trace']}")     # Shows deduction steps
```

### Example Output

```
Question: Is Benson in charge of Mordecai?
Query: in_charge_of(benson, mordecai)

[Step 1: RAG Retrieval]
  Retrieved 15 relevant KB entries
  - Facts: 10
  - Rules: 5

[Step 2: Parse Question with LangChain]
  Parsed query: in_charge_of(benson, mordecai)

[Step 3: Build Knowledge Base]
  Knowledge base contains:
  - 38 facts
  - 5 rules

[Step 4: Backward Chaining Inference]
Goal: in_charge_of(benson, mordecai)
  Trying rule: in_charge_of(?Boss_1, ?Worker_1) :- reports_to(?Worker_1, ?Boss_1)
  Goal: reports_to(mordecai, benson)
    ✓ Matched fact: reports_to(mordecai, benson)
  ✓ Rule succeeded: in_charge_of(?Boss_1, ?Worker_1) :- reports_to(?Worker_1, ?Boss_1)

[Step 5: Result]
  Query: in_charge_of(benson, mordecai)
  Result: True

✓ RESULT: True
```

## Knowledge Base

The system uses the **Regular Show** knowledge base from Task 4, which includes:

### Facts
- `park_worker(mordecai)`, `park_worker(rigby)`, etc.
- `boss(benson)`
- `park_manager(pops)`
- `friends(mordecai, rigby)`
- `reports_to(mordecai, benson)`, etc.
- `character_type(mordecai, blue_jay)`, etc.

### Rules
- `in_charge_of(Boss, Worker) :- reports_to(Worker, Boss)`
- `work_together(X, Y) :- park_worker(X), park_worker(Y), X \= Y`
- `has_authority(X) :- boss(X)`
- `has_authority(X) :- park_manager(X)`
- `is_subordinate(X) :- reports_to(X, _)`

## Example Queries

- "Is Benson in charge of Mordecai?" → **TRUE**
- "Are Mordecai and Rigby friends?" → **TRUE**
- "Is Mordecai a park worker?" → **TRUE**
- "Does Benson have authority?" → **TRUE**
- "Is Pops a subordinate?" → **FALSE*

