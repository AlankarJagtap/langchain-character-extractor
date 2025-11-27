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

    if text.startswith("```"):
        text = text.split("```", 1)[1].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# -----------------------------------------------------------
#  PROMPT FOR CHARACTER INFO EXTRACTION (STRICT)
# -----------------------------------------------------------
def build_extraction_prompt(character_name: str, story_context: str, story_title: str) -> str:
    return f"""
You are a strict information extraction system.

The story title is: "{story_title}".

Your task is to determine whether "{character_name}" is a human CHARACTER in this story.

### VERY IMPORTANT RULES
1. A character MUST be a human person in the story.
2. DO NOT treat places, buildings, concepts, events, animals, objects, or locations as characters.
3. If "{character_name}" is NOT a human character in the story, return ONLY:

{{
  "error": "Not a character in the story."
}}

4. If it IS a character, return STRICT JSON:

{{
  "name": "{character_name}",
  "storyTitle": "{story_title}",
  "summary": string,
  "relations": [
      {{ "name": string, "relation": string }}
  ],
  "characterType": "protagonist" | "antagonist" | "side character" | "unknown"
}}

No explanations. Only JSON.

Story context:
\"\"\"
{story_context}
\"\"\"
"""


# -----------------------------------------------------------
#  CORE FUNCTION — SEARCH → VALIDATE → EXTRACT JSON
# -----------------------------------------------------------
def get_character_info(character_name: str, persist_dir: str = "chroma_db"):
    """Retrieves story context, validates character, extracts structured details."""

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY not set in .env")

    # ---- 1. Load embeddings + vector DB ----
    print("🔍 Loading Chroma DB...")
    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # ---- Fast rejection of obvious non-human terms ----
    non_char_nouns = {
        "school", "village", "town", "city", "river", "road", "forest",
        "garden", "house", "building", "church", "hospital", "tree", "market"
    }

    if character_name.lower() in non_char_nouns:
        return {"error": f"'{character_name}' is not a character in the story."}

    # ---- 2. Similarity search ----
    print(f"🔎 Searching for character: {character_name}")
    results = vectorstore.similarity_search(character_name, k=4)

    if not results:
        return {"error": f"Character '{character_name}' not found in any story."}

    # ---- 3. Combine chunks & validate ----
    story_context = "\n\n".join([doc.page_content for doc in results])

    if character_name.lower() not in story_context.lower():
        return {"error": f"Character '{character_name}' not found in any story."}

    # ---- Extract story title from metadata ----
    story_titles = [doc.metadata.get("story_title", "") for doc in results]
    story_title = story_titles[0] if story_titles else "Unknown"

    # ---- 4. Build strict prompt ----
    prompt = build_extraction_prompt(character_name, story_context, story_title)

    # ---- 5. LLM Processing ----
    print("🤖 Sending to Mistral LLM...")
    llm = ChatMistralAI(model="open-mistral-7b", max_tokens=500)
    response = llm.invoke(prompt)

    raw_output = response.content.strip()
    cleaned = clean_json_fences(raw_output)

    # ---- 6. Parse JSON ----
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
    print(json.dumps(result, indent=2, ensure_ascii=False))

