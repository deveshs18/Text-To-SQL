"""Test script to verify Qwen server is working."""
import requests
import json

def test_qwen_server():
    """Test if Qwen server is responding."""
    url = "http://localhost:11438/api/generate"
    
    test_prompt = "SCHEMA: adult_income(age, income) Q: How many people? SQL:"
    
    payload = {
        "model": "qwen-0.5b-spider",
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    print("="*80)
    print("TESTING QWEN SERVER")
    print("="*80)
    print(f"URL: {url}")
    print(f"Prompt: {test_prompt}")
    print()
    
    try:
        print("Sending request...")
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Server is working!")
            print(f"Response: {result.get('response', 'No response')}")
            print(f"Done: {result.get('done', False)}")
            return True
        else:
            print(f"❌ Server returned error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server on port 11438")
        print("   Make sure the server is running:")
        print("   python text2sql/scripts/qwen_server.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out (server may be slow or stuck)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen_server()
    if success:
        print("\n✅ Server is ready for Streamlit!")
    else:
        print("\n❌ Server is not ready. Fix the issues above before using Streamlit.")


