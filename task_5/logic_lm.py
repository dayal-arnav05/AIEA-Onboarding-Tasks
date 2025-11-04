#!/usr/bin/env python3
"""
Logic-LM: Empowering Large Language Models with Symbolic Solvers
Based on the paper: https://arxiv.org/abs/2305.12295

This implementation integrates LLMs with Prolog (SWI-Prolog via pyswip) 
to perform faithful logical reasoning on the Regular Show knowledge base.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from pyswip import Prolog
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from root .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)


class LogicLM:
    """
    Logic-LM: Integrates LLM with symbolic solver (Prolog) for logical reasoning.
    
    Pipeline:
    1. Natural language question -> LLM -> Prolog query
    2. Execute Prolog query on knowledge base
    3. If error, self-refine using error messages
    4. Return natural language answer
    """
    
    def __init__(self, kb_path: str, model: str = "gpt-4", max_refinements: int = 3):
        """
        Initialize Logic-LM system.
        
        Args:
            kb_path: Path to Prolog knowledge base file
            model: OpenAI model to use
            max_refinements: Maximum number of self-refinement attempts
        """
        self.kb_path = os.path.abspath(kb_path)
        self.model = model
        self.max_refinements = max_refinements
        
        # Initialize Prolog solver
        self.prolog = Prolog()
        self._load_knowledge_base()
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Track reasoning history
        self.history: List[Dict] = []
    
    def _load_knowledge_base(self):
        """Load the Prolog knowledge base."""
        try:
            self.prolog.consult(self.kb_path)
            print(f"✓ Loaded knowledge base: {self.kb_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base: {e}")
    
    def _get_kb_schema(self) -> str:
        """Get the schema/structure of the knowledge base for LLM context."""
        return """
Knowledge Base Schema (Regular Show):

Facts:
- park_worker(Person): Person is a park worker
- boss(Person): Person is a boss
- park_manager(Person): Person is a park manager
- character_type(Person, Type): Person is of Type
- friends(Person1, Person2): Person1 and Person2 are friends
- reports_to(Worker, Boss): Worker reports to Boss

Rules:
- in_charge_of(Boss, Worker): Boss is in charge of Worker if Worker reports to Boss
- work_together(X, Y): X and Y work together if both are park workers
- has_authority(X): X has authority if X is boss or park manager
- is_subordinate(X): X is subordinate if X reports to someone

Available individuals:
- mordecai, rigby, skips, muscle_man, hi_five_ghost (park workers)
- benson (boss)
- pops (park manager)
"""
    
    def translate_to_prolog(self, question: str, error_msg: Optional[str] = None) -> str:
        """
        Use LLM to translate natural language question to Prolog query.
        
        Args:
            question: Natural language question
            error_msg: Optional error message from previous attempt (for refinement)
            
        Returns:
            Prolog query string
        """
        system_prompt = f"""You are an expert at translating natural language questions into Prolog queries.

{self._get_kb_schema()}

Guidelines:
1. Return ONLY the Prolog query, nothing else
2. Use lowercase for atoms (mordecai, not Mordecai)
3. Use uppercase for variables (X, Y, Boss, Worker)
4. Use underscore _ for anonymous variables
5. Common query patterns:
   - Find all: predicate(X)
   - Check specific: predicate(atom1, atom2)
   - Count: findall(X, predicate(X), List)
   - Negation: \\+ predicate(X)

Examples:
Q: "Who are the park workers?"
A: park_worker(X)

Q: "Is Mordecai friends with Rigby?"
A: friends(mordecai, rigby)

Q: "Who does Benson manage?"
A: in_charge_of(benson, X)

Q: "Does Mordecai report to anyone?"
A: reports_to(mordecai, X)
"""
        
        if error_msg:
            system_prompt += f"""

PREVIOUS ATTEMPT FAILED WITH ERROR:
{error_msg}

Please revise the query to fix this error. Common fixes:
- Check atom spelling and use lowercase
- Ensure proper variable naming (uppercase)
- Verify predicate exists in schema
- Check argument order
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.0
            )
            
            query = response.choices[0].message.content.strip()
            # Remove any markdown code blocks
            query = query.replace("```prolog", "").replace("```", "").strip()
            
            return query
            
        except Exception as e:
            raise RuntimeError(f"LLM translation failed: {e}")
    
    def execute_query(self, query: str) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Execute Prolog query on the knowledge base.
        
        Args:
            query: Prolog query string
            
        Returns:
            (success, results, error_message)
        """
        try:
            results = list(self.prolog.query(query))
            return True, results, None
        except Exception as e:
            error_msg = str(e)
            return False, [], error_msg
    
    def format_results(self, question: str, query: str, results: List[Dict]) -> str:
        """
        Use LLM to format Prolog results into natural language answer.
        
        Args:
            question: Original natural language question
            query: Prolog query that was executed
            results: Results from Prolog query
            
        Returns:
            Natural language answer
        """
        system_prompt = """You are an expert at interpreting Prolog query results and formatting them as natural language answers.

Guidelines:
1. Provide a clear, concise answer to the question
2. If results are empty, say "No" or "None found"
3. If results contain variables, list them clearly
4. For yes/no questions, answer clearly
5. Be conversational and natural
"""
        
        user_prompt = f"""Question: {question}

Prolog Query: {query}

Query Results: {results}

Please provide a natural language answer to the question based on these results."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # Fallback to simple formatting
            if not results:
                return "No results found."
            return str(results)
    
    def reason(self, question: str, verbose: bool = True) -> Dict:
        """
        Main reasoning pipeline: question -> Prolog query -> answer
        
        Args:
            question: Natural language question
            verbose: Whether to print intermediate steps
            
        Returns:
            Dictionary with reasoning trace and answer
        """
        if verbose:
            print("\n" + "="*80)
            print(f"Question: {question}")
            print("="*80)
        
        reasoning_trace = {
            "question": question,
            "attempts": [],
            "final_answer": None,
            "success": False
        }
        
        error_msg = None
        
        # Self-refinement loopu7≥
        for attempt in range(self.max_refinements):
            if verbose:
                print(f"\n[Attempt {attempt + 1}/{self.max_refinements}]")
            
            # Step 1: Translate to Prolog
            if verbose:
                print("Translating to Prolog...")
            
            prolog_query = self.translate_to_prolog(question, error_msg)
            
            if verbose:
                print(f"  Prolog Query: {prolog_query}")
            
            # Step 2: Execute query
            if verbose:
                print("Executing query...")
            
            success, results, error_msg = self.execute_query(prolog_query)
            
            attempt_data = {
                "attempt_num": attempt + 1,
                "prolog_query": prolog_query,
                "success": success,
                "results": results,
                "error": error_msg
            }
            reasoning_trace["attempts"].append(attempt_data)
            
            if success:
                # Step 3: Format answer
                if verbose:
                    print(f"  Results: {results}")
                    print("Formatting answer...")
                
                answer = self.format_results(question, prolog_query, results)
                
                reasoning_trace["final_answer"] = answer
                reasoning_trace["success"] = True
                
                if verbose:
                    print(f"\n✓ Answer: {answer}")
                
                # Store in history
                self.history.append(reasoning_trace)
                return reasoning_trace
            
            else:
                # Query failed, prepare for refinement
                if verbose:
                    print(f"  ✗ Error: {error_msg}")
                    if attempt < self.max_refinements - 1:
                        print("  Attempting self-refinement...")
        
        # All attempts failed
        reasoning_trace["final_answer"] = f"Failed to answer after {self.max_refinements} attempts."
        if verbose:
            print(f"\n✗ {reasoning_trace['final_answer']}")
        
        self.history.append(reasoning_trace)
        return reasoning_trace
    
    def interactive_mode(self):
        """Run Logic-LM in interactive mode."""
        print("\n" + "="*80)
        print("Logic-LM Interactive Mode")
        print("Ask questions about the Regular Show knowledge base")
        print("Type 'quit' or 'exit' to stop")
        print("="*80)
        
        while True:
            try:
                question = input("\n> ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                self.reason(question, verbose=True)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Main function to demonstrate Logic-LM."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Path to Regular Show knowledge base
    kb_path = "../task_4/regular_show_kb.pl"
    
    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base not found at {kb_path}")
        sys.exit(1)
    
    # Initialize Logic-LM
    print("\nInitializing Logic-LM...")
    logic_lm = LogicLM(kb_path=kb_path, model="gpt-4")
    
    # Demo questions
    demo_questions = [
        "Who are all the park workers?",
        "Is Mordecai friends with Rigby?",
        "Who does Benson manage?",
        "What type of character is Mordecai?",
        "Who has authority in the park?",
        "Does Rigby report to anyone?",
    ]
    
    print("\n" + "="*80)
    print("Running Demo Questions")
    print("="*80)
    
    for question in demo_questions:
        logic_lm.reason(question, verbose=True)
    
    # Interactive mode
    print("\n" + "="*80)
    print("Starting Interactive Mode...")
    print("="*80)
    logic_lm.interactive_mode()


if __name__ == "__main__":
    main()

