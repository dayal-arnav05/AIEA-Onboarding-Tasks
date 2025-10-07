import os 
import subprocess
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = '../.env'

load_dotenv(env_path)
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=api_key, timeout=30.0)

def validate_prolog_syntax(prolog_code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
        f.write(prolog_code)
        temp_file = f.name
    
    result = subprocess.run(
        ['swipl', '-q', '-t', 'halt', temp_file],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    os.unlink(temp_file)
    return result.returncode == 0

def generate_and_validate_prolog():
    try:
        print("API Client Connected")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Prolog expert. Generate only valid Prolog code."},
                {"role": "user", "content": "Create a knowledge base with prolog. Please keep your text output strictly prolog code we will run your KB through SWI-Prolog. Please make it short."}
            ],
            max_tokens=200,
            timeout=60
        )
        
        prolog_code = response.choices[0].message.content
        print(prolog_code)
        is_valid = validate_prolog_syntax(prolog_code)
        print(f"Prolog validation result: {is_valid}")
        
        return prolog_code, is_valid
        
    except Exception as e:
        print(f"API call failed: {type(e).__name__}: {e}")
        return None, False

if __name__ == "__main__":
    prolog_code, is_valid = generate_and_validate_prolog()


