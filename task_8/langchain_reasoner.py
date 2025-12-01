"""
LangChain Reasoner: Integrates RAG, LangChain, and Backward Chaining.

This is the main orchestration module that combines:
- RAG (ChromaDB) for knowledge retrieval
- LangChain for natural language processing
- Backward Chaining for logical inference

Complete Pipeline:
1. Natural Language Question → RAG retrieves relevant KB entries
2. LangChain parses question → Prolog query format
3. Build dynamic Knowledge Base from retrieved entries
4. Backward Chaining proves query with trace
5. Return TRUE/FALSE with logical deduction steps

Example:
    reasoner = LangChainReasoner()
    result = reasoner.reason("Is Benson in charge of Mordecai?")
    # Returns: {"result": True, "trace": "Goal: in_charge_of...", ...}

Critical for:
- End-to-end neuro-symbolic AI pipeline
- Natural language understanding + formal reasoning
- Explainable AI (logical trace shows deduction steps)
"""

import os
import sys
from typing import Dict, List, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence

# Import our components
from kb_to_chromadb import query_kb
from prolog_parser import parse_prolog_line

# Import backward chainer from task_7
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'task_7'))
from backward_chain import Fact, Rule, KnowledgeBase, BackwardChainer

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)


class LangChainReasoner:
    """
    LangChain-powered reasoning system that combines:
    - RAG (ChromaDB) for knowledge retrieval
    - LangChain for NL processing
    - Backward Chaining for logical inference
    """
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initialize the LangChain reasoner.
        
        Args:
            model: OpenAI model to use
            temperature: Temperature for LLM
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self._build_chains()
    
    def _build_chains(self):
        """Build LangChain processing chains."""
        
        # Chain 1: Parse natural language question to Prolog fact
        parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting natural language questions into Prolog-style predicate queries.

Given a question about the Regular Show knowledge base, convert it to a simple predicate format.

Available predicates:
- park_worker(X): X is a park worker
- boss(X): X is a boss
- park_manager(X): X is a park manager
- character_type(X, Type): X is of type Type
- friends(X, Y): X and Y are friends
- reports_to(X, Y): X reports to Y
- in_charge_of(X, Y): X is in charge of Y
- work_together(X, Y): X and Y work together
- has_authority(X): X has authority
- is_subordinate(X): X is subordinate

Rules:
- Use lowercase for specific entities (mordecai, benson, rigby)
- Use uppercase for variables (X, Y)
- Return ONLY the predicate, nothing else

Examples:
Q: "Is Benson in charge of Mordecai?"
A: in_charge_of(benson, mordecai)

Q: "Are Mordecai and Rigby friends?"
A: friends(mordecai, rigby)

Q: "Who does Benson manage?"
A: in_charge_of(benson, X)

Q: "Is Mordecai a park worker?"
A: park_worker(mordecai)"""),
            ("user", "{question}")
        ])
        
        self.parse_chain = parse_prompt | self.llm
        
        # Chain 2: Format the final answer with trace
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at explaining logical reasoning results.

Given a query result (TRUE/FALSE) and a reasoning trace, provide a clear, concise summary.

Keep it brief and natural. Include the key deduction steps."""),
            ("user", """Question: {question}
Query: {query}
Result: {result}
Trace: {trace}

Provide a natural language explanation of the result and reasoning.""")
        ])
        
        self.format_chain = format_prompt | self.llm
    
    def parse_question_to_fact(self, question: str, kb_context: str = "") -> Optional[Fact]:
        """
        Parse natural language question to Fact object using LLM.
        
        Args:
            question: Natural language question
            kb_context: Retrieved KB context
            
        Returns:
            Fact object or None
        """
        try:
            response = self.parse_chain.invoke({"question": question})
            prolog_query = response.content.strip()
            
            # Remove any markdown formatting
            prolog_query = prolog_query.replace("```", "").strip()
            
            print(f"  Parsed query: {prolog_query}")
            
            # Parse the Prolog query to a Fact
            from prolog_parser import parse_prolog_fact
            fact = parse_prolog_fact(prolog_query)
            
            return fact
            
        except Exception as e:
            print(f"  Error parsing question: {e}")
            return None
    
    def build_kb_from_retrieved(self, retrieved_docs: List[Dict]) -> KnowledgeBase:
        """
        Build a KnowledgeBase from retrieved documents.
        
        Args:
            retrieved_docs: List of documents from ChromaDB
            
        Returns:
            KnowledgeBase object
        """
        kb = KnowledgeBase()
        
        for doc in retrieved_docs:
            prolog_str = doc['prolog']
            doc_type = doc['type']
            
            result = parse_prolog_line(prolog_str)
            if result:
                kind, obj = result
                if kind == "fact":
                    kb.add_fact(obj)
                elif kind == "rule":
                    kb.add_rule(obj)
        
        return kb
    
    def reason(self, question: str, verbose: bool = True) -> Dict:
        """
        Main reasoning pipeline using LangChain and backward chaining.
        
        Args:
            question: Natural language question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with result and trace
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"Question: {question}")
            print("=" * 80)
        
        # Step 1: RAG - Retrieve relevant KB entries
        if verbose:
            print("\n[Step 1: RAG Retrieval]")
        
        retrieved_docs = query_kb(question, n_results=15)
        
        if verbose:
            print(f"  Retrieved {len(retrieved_docs)} relevant KB entries")
            print(f"  - Facts: {sum(1 for d in retrieved_docs if d['type'] == 'fact')}")
            print(f"  - Rules: {sum(1 for d in retrieved_docs if d['type'] == 'rule')}")
        
        # Step 2: LangChain - Parse question to Fact
        if verbose:
            print("\n[Step 2: Parse Question with LangChain]")
        
        fact = self.parse_question_to_fact(question)
        
        if not fact:
            return {
                "question": question,
                "result": None,
                "success": False,
                "error": "Failed to parse question"
            }
        
        # Step 3: Build KB from retrieved documents
        if verbose:
            print("\n[Step 3: Build Knowledge Base]")
        
        kb = self.build_kb_from_retrieved(retrieved_docs)
        
        if verbose:
            print(f"  Knowledge base contains:")
            print(f"  - {len(kb.facts)} facts")
            print(f"  - {len(kb.rules)} rules")
        
        # Step 4: Run Backward Chainer with trace
        if verbose:
            print("\n[Step 4: Backward Chaining Inference]")
        
        chainer = BackwardChainer(kb, trace=verbose, max_depth=50)
        
        # Capture the trace output
        import io
        from contextlib import redirect_stdout
        
        trace_buffer = io.StringIO()
        with redirect_stdout(trace_buffer):
            result = chainer.prove(fact)
        
        trace = trace_buffer.getvalue()
        
        # Step 5: Format result
        if verbose:
            print(f"\n[Step 5: Result]")
            print(f"  Query: {fact}")
            print(f"  Result: {result}")
        
        return {
            "question": question,
            "query": str(fact),
            "result": result,
            "success": True,
            "trace": trace,
            "retrieved_kb": [doc['prolog'] for doc in retrieved_docs],
            "kb_stats": {
                "facts": len(kb.facts),
                "rules": len(kb.rules)
            }
        }
    
    def explain_result(self, reasoning_result: Dict) -> str:
        """
        Generate natural language explanation of the reasoning result.
        
        Args:
            reasoning_result: Result from reason() method
            
        Returns:
            Natural language explanation
        """
        try:
            response = self.format_chain.invoke({
                "question": reasoning_result["question"],
                "query": reasoning_result["query"],
                "result": reasoning_result["result"],
                "trace": reasoning_result.get("trace", "")
            })
            return response.content.strip()
        except Exception as e:
            return f"Result: {reasoning_result['result']}"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("Set it in your environment or .env file")
        sys.exit(1)
    
    # Initialize reasoner
    print("Initializing LangChain Reasoner...")
    reasoner = LangChainReasoner()
    print("✓ Ready\n")
    
    # Test query
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Is Benson in charge of Mordecai?"
    
    print(f"Query: {question}\n")
    result = reasoner.reason(question, verbose=True)
    
    print("\n" + "=" * 80)
    print(f"RESULT: {result['result']}")
    print("=" * 80)

