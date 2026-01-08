"""
Quick test script to verify your setup
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

print("Testing AI Resume Analyzer setup...\n")

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env file")
    print("   Create a .env file with: OPENAI_API_KEY=your-key-here")
    exit(1)
else:
    print("✅ OPENAI_API_KEY found")

# Test LLM connection
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    response = llm.invoke("Say 'Setup successful!' if you can read this.")
    print("✅ OpenAI connection successful")
    print(f"   Response: {response.content}")
except Exception as e:
    print(f"❌ OpenAI connection failed: {e}")
    exit(1)

print("\n🎉 All checks passed! Run 'streamlit run app.py' to start.")
