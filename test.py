import os
from dotenv import load_dotenv
from openai import OpenAI

# loading environment variables from the .env file
load_dotenv()

# setting api key and base url
api_key = os.getenv("AIML_API_KEY")
base_url = "https://api.aimlapi.com/v1"

# A robust check to ensure the API key is set
if not api_key:
    raise ValueError("API key not found. Please set the AIML_API_KEY environment variable or create a .env file.")

print("api-key is ", api_key)

system_prompt = "You are a travel agent. Be descriptive and helpful."
user_prompt = "Tell me about Zürich"

api = OpenAI(api_key=api_key, base_url=base_url)


def main():
    completion = api.chat.completions.create(
        model="google/gemma-3-4b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    response = completion.choices[0].message.content

    print("User:", user_prompt)
    print("AI:", response)


if __name__ == "__main__":
    main()