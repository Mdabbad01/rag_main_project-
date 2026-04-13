import time
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL


def get_client():
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=GROQ_API_KEY)


def generate_response(prompt: str) -> str:
    client = get_client()

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()

            return "I could not generate a response."

        except Exception as e:
            last_error = str(e)

            # Rate limit / temporary overload
            if "429" in last_error or "rate_limit" in last_error.lower():
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return "The Groq API is temporarily rate-limited. Please try again in a few moments."

            # Other errors
            return f"LLM error: {last_error}"

    return f"LLM error: {last_error}"