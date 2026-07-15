import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

# Safely pluck the key out of memory
api_key = os.getenv("OPENROUTER_API_KEY")

print(f"2. Checking API key value...")
if not api_key:
    print("OPENROUTER_API_KEY returned None! Checking your .env file layout.")
    # Fallback to prevent the validation wrapper from throwing an initialization crash
    api_key = "dummy-key-for-validation"

print("3. Spawning the ChatOpenAI client wrapper...")
llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=api_key, 
    base_url="https://openrouter.ai/api/v1"
)

print("4. Attempting to dispatch the network request to OpenRouter...")
try:
    response = llm.invoke("Say hello")
    print("SUCCESS! Response received:")
    print(response.content)
except Exception as e:
    print(f"\nTHE REMOTE SERVER REJECTED THE CALL. Details:\n{e}")