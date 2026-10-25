import os

from dotenv import load_dotenv
from google import genai


def main():
    # Load environment variables from .env
    load_dotenv()

    # Read the API key without hardcoding it
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Check your .env file."
        )

    # Create the Gemini API client
    client = genai.Client(api_key=api_key)

    try:
        # Send a simple prompt to Gemini
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Reply with exactly: Gemini connection successful.",
        )

        # Display the model's response
        if response.text:
            print(response.text)
        else:
            print("The API returned no text.")

    except Exception as exc:
        print(
            f"Gemini request failed: {type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    main()