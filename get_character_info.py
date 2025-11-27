# get_character_info.py

import os
import json
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# -----------------------------------------------------------
#  CLEAN JSON FENCE STRINGS FROM LLM OUTPUT
# -----------------------------------------------------------
def clean_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    text = text.strip()

    # If starts with ```
    if text.startswith("```"):
        # Remove the first ```
        text = text.split("```", 1)[1].strip()

        # If it starts with json or JSON
        if text.lower().startswith("json"):
            # remove the word "json"
            text = text[4:].strip()

    # If ends with ```
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# -----------------------------------------------------------
#  PROMPT FOR CHARACTER INFO EXTRACTION
# -----------------------------------------------------------
def build_extraction_prompt(character_name: str, story_context: str) -> str:
    return f"""
You are an information extraction assistant.

Extract information for the character "{character_name}" from the provided story context.

Return STRICT JSON with the following keys:

- name
- storyTitle
- summary
- relations: an array of objects → {{ "name": string, "relation": string }}
- characterType: protagonist, antagonist, side character, unknown

If information is missing, return empty strings or an empty list.

OUTPUT ONLY JSON. No explanation.

Story context:
\"\"\"
{story_context}
\"\"\"
"""


# -----------------------------------------------------------
#  CORE FUNCTION — SEARCH → EXTRACT → RETURN JSON
# -----------------------------------------------------------
def get_character_info(character_name: str, persist_dir: str = "chroma_db"):
    """Query Chroma, retrieve story chunks, send to Mistral for structured extraction."""

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    # ---- 1. Load embeddings + Chroma ----
    print("🔍 Loading Chroma DB...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # ---- 2. Search for character ----
    print(f"🔎 Searching for character: {character_name}")
    results = vectorstore.similarity_search(character_name, k=4)

    if not results:
        return {"error": f"Character '{character_name}' not found in any story."}

    # Combine all retrieved chunks
    story_context = "\n\n".join([doc.page_content for doc in results])

    # ---- 3. Build prompt ----
    prompt = build_extraction_prompt(character_name, story_context)

    # ---- 4. Mistral LLM ----
    print("🤖 Sending to Mistral LLM...")
    llm = ChatMistralAI(model="open-mistral-7b")
    response = llm.invoke(prompt)

    raw_output = response.content.strip()
    cleaned = clean_json_fences(raw_output)

    # ---- 5. Parse JSON ----
    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        return {
            "error": "LLM returned invalid JSON.",
            "raw_output": raw_output,
            "cleaned_attempt": cleaned,
        }


# -----------------------------------------------------------
#  CLI WRAPPER
# -----------------------------------------------------------
def get_character_info_cli(character_name: str, persist_dir: str):
    result = get_character_info(character_name, persist_dir)
    print(json.dumps(result, indent=2))
