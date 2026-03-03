
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model = "deepseek/deepseek-v3.2"

print(f"URL: {base_url}/chat/completions")
print(f"Key: {api_key[:10] if api_key else '(not set)'}...")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://test.com",
    "X-Title": "Test Script"
}

data = {
    "model": model,
    "messages": [{"role": "user", "content": "Hello!"}],
}

try:
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
