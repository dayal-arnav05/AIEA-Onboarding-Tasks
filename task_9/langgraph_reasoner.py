"""
LangGraph Reasoner: State machine-based reasoning with relevancy judgment and self-refinement.

This module uses LangGraph to create a stateful reasoning pipeline with:
- RAG retrieval from ChromaDB
- Relevancy judgment of retrieved documents
- Self-refinement through additional retrieval or Chain of Thought
- Backward chaining inference
- Explainable traces

LangGraph State Machine:
1. retrieve → Judge relevancy
2. If low relevancy → refine (additional retrieval)
3. parse → build_kb → infer → format
4. Return result with trace

Key improvements over Task 8:
- Stateful execution with LangGraph
- Automatic relevancy checking
- Self-refinement loop
- More robust error handling
"""

import os
import sys
from typing import Dict, List, TypedDict, Annotated
from dotenv import load_dotenv
import operator

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Import from task_8
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task_8'))
import kb_to_chromadb
from prolog_parser import parse_prolog_line, parse_prolog_fact

# Wrapper to use task_8's ChromaDB
def query_kb(query: str, n_results: int = 15):
    """Query ChromaDB from task_8."""
    import chromadb
    db_path = os.path.join(os.path.dirname(__file__), '..', 'task_8', 'chroma_db')
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("regular_show_kb")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    documents = []
    for i in range(len(results['documents'][0])):
        documents.append({
            "text": results['documents'][0][i],
            "prolog": results['metadatas'][0][i]['prolog'],
            "predicate": results['metadatas'][0][i]['predicate'],
            "type": results['metadatas'][0][i]['type']
        })
    
    return documents

# Import backward chainer from task_7
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task_7'))
from backward_chain import Fact, Rule, KnowledgeBase, BackwardChainer

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)


# State definition for LangGraph
class ReasoningState(TypedDict):
    """State object that flows through the LangGraph."""
    question: str
    retrieved_docs: List[Dict]
    relevancy_score: float
    relevancy_explanation: str
    needs_refinement: bool
    refinement_query: str
    parsed_fact: Fact
    knowledge_base: KnowledgeBase
    inference_result: bool
    trace: str
    final_answer: str
    iteration: int
    max_iterations: int


class LangGraphReasoner:
    """
    LangGraph-based reasoning system with relevancy judgment and self-refinement.
    
    Pipeline:
    1. retrieve: RAG retrieval from ChromaDB
    2. judge_relevancy: LLM judges if retrieved docs are relevant
    3. refine (conditional): If low relevancy, get more context
    4. parse: Convert NL question to Fact
    5. build_kb: Construct KnowledgeBase from docs
    6. infer: Backward chaining inference
    7. format: Generate final answer
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """Initialize the LangGraph reasoner."""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        # Create graph with state
        workflow = StateGraph(ReasoningState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("judge_relevancy", self.judge_relevancy_node)
        workflow.add_node("refine", self.refine_node)
        workflow.add_node("parse", self.parse_node)
        workflow.add_node("build_kb", self.build_kb_node)
        workflow.add_node("infer", self.infer_node)
        workflow.add_node("format", self.format_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Add edges
        workflow.add_edge("retrieve", "judge_relevancy")
        
        # Conditional edge: refine if low relevancy
        workflow.add_conditional_edges(
            "judge_relevancy",
            self.should_refine,
            {
                "refine": "refine",
                "parse": "parse"
            }
        )
        
        workflow.add_edge("refine", "judge_relevancy")  # Re-judge after refinement
        workflow.add_edge("parse", "build_kb")
        workflow.add_edge("build_kb", "infer")
        workflow.add_edge("infer", "format")
        workflow.add_edge("format", END)
        
        return workflow.compile()
    
    def should_refine(self, state: ReasoningState) -> str:
        """
        Decide if we need to refine the retrieval.
        
        Logic:
        - If relevancy < 0.7 AND iterations < max: refine
        - Otherwise: proceed to parse
        """
        if state["needs_refinement"] and state["iteration"] < state["max_iterations"]:
            return "refine"
        return "parse"
    
    def retrieve_node(self, state: ReasoningState) -> Dict:
        """Node 1: Retrieve relevant documents from ChromaDB."""
        print(f"\n[Node: Retrieve] Query: {state['question']}")
        
        # If this is a refinement iteration, use the refinement query
        query = state.get("refinement_query", state["question"])
        
        # Retrieve from ChromaDB
        docs = query_kb(query, n_results=15)
        
        print(f"  Retrieved {len(docs)} documents")
        print(f"  - Facts: {sum(1 for d in docs if d['type'] == 'fact')}")
        print(f"  - Rules: {sum(1 for d in docs if d['type'] == 'rule')}")
        
        return {
            "retrieved_docs": docs,
            "iteration": state.get("iteration", 0)
        }
    
    def judge_relevancy_node(self, state: ReasoningState) -> Dict:
        """Node 2: Judge relevancy of retrieved documents using LLM."""
        print(f"\n[Node: Judge Relevancy]")
        
        # Create prompt for relevancy judgment
        relevancy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at judging document relevancy for logical reasoning tasks.

Given a question and retrieved documents, judge if the documents contain enough relevant information to answer the question.

**CRITICAL**: For questions that require inference (e.g., "Is X in charge of Y?"), you MUST check:
1. Are the key entities mentioned? (e.g., Benson, Mordecai)
2. Are direct facts present? (e.g., reports_to relationships)
3. **Are inference RULES present?** (e.g., in_charge_of rules)

If the question asks about a derived relationship (in_charge_of, has_authority, work_together, etc.) 
but NO RULES are retrieved, score MUST be < 0.7 even if facts are present.

Rules are identified by " :-" syntax: "predicate(X,Y) :- condition1, condition2"

Scoring:
- 0.9-1.0: All necessary facts AND rules present
- 0.7-0.9: Facts present, rules may help but not essential
- 0.4-0.7: Facts present but missing critical rules
- 0.0-0.4: Missing key information

Respond with:
SCORE: <float>
EXPLANATION: <why this score, mention if rules are missing>
REFINEMENT: <suggested query or "none">"""),
            ("user", """Question: {question}

Retrieved Documents:
{docs}

Count:
- Facts: {fact_count}
- Rules: {rule_count}

Judge the relevancy:""")
        ])
        
        # Format docs for prompt
        docs_str = "\n".join([
            f"{i+1}. [{d['type']}] {d['prolog']}"
            for i, d in enumerate(state["retrieved_docs"][:10])
        ])
        
        # Count facts and rules
        fact_count = sum(1 for d in state["retrieved_docs"] if d['type'] == 'fact')
        rule_count = sum(1 for d in state["retrieved_docs"] if d['type'] == 'rule')
        
        # Get LLM judgment
        chain = relevancy_prompt | self.llm
        response = chain.invoke({
            "question": state["question"],
            "docs": docs_str,
            "fact_count": fact_count,
            "rule_count": rule_count
        })
        
        # Parse response
        response_text = response.content
        lines = response_text.strip().split('\n')
        
        score = 0.8  # Default
        explanation = "Documents appear relevant"
        refinement = "none"
        
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split("SCORE:")[1].strip())
                except:
                    pass
            elif line.startswith("EXPLANATION:"):
                explanation = line.split("EXPLANATION:")[1].strip()
            elif line.startswith("REFINEMENT:"):
                refinement = line.split("REFINEMENT:")[1].strip()
        
        needs_refinement = score < 0.7 and refinement.lower() != "none"
        
        print(f"  Relevancy Score: {score:.2f}")
        print(f"  Explanation: {explanation}")
        if needs_refinement:
            print(f"  Low relevancy - refinement needed")
            print(f"  Refinement query: {refinement}")
        else:
            print(f"  ✓ Relevancy sufficient")
        
        return {
            "relevancy_score": score,
            "relevancy_explanation": explanation,
            "needs_refinement": needs_refinement,
            "refinement_query": refinement if needs_refinement else state["question"]
        }
    
    def refine_node(self, state: ReasoningState) -> Dict:
        """Node 3: Refine retrieval with additional context."""
        print(f"\n[Node: Refine] Iteration {state['iteration'] + 1}")
        print(f"  Using refined query: {state['refinement_query']}")
        
        # Retrieve with refined query
        additional_docs = query_kb(state['refinement_query'], n_results=10)
        
        # Merge with existing docs (avoid duplicates)
        existing_prolog = {d['prolog'] for d in state['retrieved_docs']}
        new_docs = [d for d in additional_docs if d['prolog'] not in existing_prolog]
        
        combined_docs = state['retrieved_docs'] + new_docs
        
        print(f"  Added {len(new_docs)} new documents")
        print(f"  Total documents: {len(combined_docs)}")
        
        return {
            "retrieved_docs": combined_docs,
            "iteration": state["iteration"] + 1,
            "needs_refinement": False  # Reset for re-judgment
        }
    
    def parse_node(self, state: ReasoningState) -> Dict:
        """Node 4: Parse question to Fact using LLM."""
        print(f"\n[Node: Parse Question]")
        
        # Create parse prompt
        parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting natural language to Prolog queries.

Available predicates:
- park_worker(X), boss(X), park_manager(X)
- character_type(X, Type)
- friends(X, Y)
- reports_to(X, Y)
- in_charge_of(X, Y)
- work_together(X, Y)
- has_authority(X)
- is_subordinate(X)

Rules:
- Use lowercase for entities (benson, mordecai)
- Use uppercase for variables (X, Y)
- Return ONLY the Prolog query"""),
            ("user", "{question}")
        ])
        
        chain = parse_prompt | self.llm
        response = chain.invoke({"question": state["question"]})
        prolog_query = response.content.strip().replace("```", "")
        
        print(f"  Parsed query: {prolog_query}")
        
        # Convert to Fact
        fact = parse_prolog_fact(prolog_query)
        
        if not fact:
            print(f"  ✗ Failed to parse query")
            return {"parsed_fact": None}
        
        print(f"  ✓ Fact: {fact}")
        
        return {"parsed_fact": fact}
    
    def build_kb_node(self, state: ReasoningState) -> Dict:
        """Node 5: Build KnowledgeBase from retrieved documents."""
        print(f"\n[Node: Build Knowledge Base]")
        
        kb = KnowledgeBase()
        
        for doc in state["retrieved_docs"]:
            prolog_str = doc['prolog']
            result = parse_prolog_line(prolog_str)
            
            if result:
                kind, obj = result
                if kind == "fact":
                    kb.add_fact(obj)
                elif kind == "rule":
                    kb.add_rule(obj)
        
        print(f"  KB contains:")
        print(f"  - {len(kb.facts)} facts")
        print(f"  - {len(kb.rules)} rules")
        
        return {"knowledge_base": kb}
    
    def infer_node(self, state: ReasoningState) -> Dict:
        """Node 6: Run backward chaining inference."""
        print(f"\n[Node: Backward Chaining Inference]")
        
        if not state["parsed_fact"]:
            print(f"  ✗ No fact to prove")
            return {"inference_result": False, "trace": "Failed to parse question"}
        
        # Create backward chainer
        chainer = BackwardChainer(state["knowledge_base"], trace=True, max_depth=50)
        
        # Capture trace
        import io
        from contextlib import redirect_stdout
        
        trace_buffer = io.StringIO()
        with redirect_stdout(trace_buffer):
            result = chainer.prove(state["parsed_fact"])
        
        trace = trace_buffer.getvalue()
        
        print(f"  Query: {state['parsed_fact']}")
        print(f"  Result: {result}")
        
        return {
            "inference_result": result,
            "trace": trace
        }
    
    def format_node(self, state: ReasoningState) -> Dict:
        """Node 7: Format final answer."""
        print(f"\n[Node: Format Answer]")
        
        # Create format prompt
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at explaining logical reasoning.

Given a question, result, and trace, provide a clear natural language answer.
Be concise but mention the key deduction steps."""),
            ("user", """Question: {question}
Result: {result}
Trace: {trace}

Provide a natural language answer:""")
        ])
        
        chain = format_prompt | self.llm
        response = chain.invoke({
            "question": state["question"],
            "result": state["inference_result"],
            "trace": state["trace"]
        })
        
        final_answer = response.content.strip()
        
        print(f"  ✓ Answer generated")
        
        return {"final_answer": final_answer}
    
    def reason(self, question: str, verbose: bool = True, max_iterations: int = 3) -> Dict:
        """
        Main reasoning method using LangGraph.
        
        Args:
            question: Natural language question
            verbose: Print intermediate steps
            max_iterations: Max refinement iterations
            
        Returns:
            Complete reasoning result with trace
        """
        if verbose:
            print("\n" + "=" * 80)
            print("LANGGRAPH REASONING PIPELINE")
            print("=" * 80)
            print(f"Question: {question}")
        
        # Initialize state
        initial_state = {
            "question": question,
            "retrieved_docs": [],
            "relevancy_score": 0.0,
            "relevancy_explanation": "",
            "needs_refinement": False,
            "refinement_query": "",
            "parsed_fact": None,
            "knowledge_base": None,
            "inference_result": False,
            "trace": "",
            "final_answer": "",
            "iteration": 0,
            "max_iterations": max_iterations
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Package result
        result = {
            "question": question,
            "query": str(final_state.get("parsed_fact", "")),
            "result": final_state.get("inference_result", False),
            "trace": final_state.get("trace", ""),
            "final_answer": final_state.get("final_answer", ""),
            "relevancy_score": final_state.get("relevancy_score", 0.0),
            "relevancy_explanation": final_state.get("relevancy_explanation", ""),
            "iterations": final_state.get("iteration", 0),
            "retrieved_count": len(final_state.get("retrieved_docs", [])),
            "success": True
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print("FINAL RESULT")
            print("=" * 80)
            print(f"Result: {result['result']}")
            print(f"Relevancy Score: {result['relevancy_score']:.2f}")
            print(f"Refinement Iterations: {result['iterations']}")
            print(f"\nTrace:\n{result['trace']}")
        
        return result


if __name__ == "__main__":
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    # Initialize reasoner
    print("Initializing LangGraph Reasoner...")
    reasoner = LangGraphReasoner()
    print("✓ Ready\n")
    
    # Test query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Is Benson in charge of Mordecai?"
    
    result = reasoner.reason(question, verbose=True)
    
    print(f"\n{'=' * 80}")
    print(f"✓ RESULT: {result['result']}")
    print(f"{'=' * 80}")

