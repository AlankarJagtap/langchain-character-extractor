# 🌟 LangChain Character Extractor

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![MistralAI](https://img.shields.io/badge/MistralAI-LLM-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A fully functional **RAG (Retrieval-Augmented Generation)** pipeline that extracts **structured character information** from story documents using:

- 🧠 **MistralAI** models  
- 🔍 **Semantic Vector Search (ChromaDB)**  
- 🪄 **LangChain** orchestration  
- 📝 **Strict JSON extraction**  
- 🔒 **Anti-hallucination guardrails**  

---

# 📦 Features at a Glance

| Feature | Description |
|--------|-------------|
| 🔎 Vector Retrieval | Finds relevant story segments using embeddings |
| 📚 RAG Pipeline | Retrieval → Augmentation → LLM Generation |
| 🧱 Structured Output | Name, summary, relations, character type |
| 🤖 Mistral LLM | Extraction with strict JSON |
| 🚫 Anti-Hallucination | Rejects non-human entities and false matches |
| 🧪 Edge Case Handling | Clean errors for missing characters |
| 🛠 CLI Tools | Easy to run and test |

---

# 🧠 Why This Is a RAG System

RAG = **Retrieve + Augment + Generate**

Traditional LLMs cannot answer questions about documents they haven't seen.  
This system solves that via:

1. **Embed stories** into vector space (`mistral-embed`)
2. **Store embeddings** in ChromaDB
3. **Retrieve relevant segments** using semantic search
4. **Augment LLM input** with only the top relevant chunks
5. **Generate structured JSON** using an LLM (open-mistral-7b)

---

# 🗂 System Architecture (ASCII Diagram)

```
                   ┌────────────────────────┐
                   │        Story Files      │
                   │        (data/*.txt)     │
                   └─────────────┬──────────┘
                                 │
                                 ▼
                   ┌────────────────────────┐
                   │   Text Splitter         │
                   │ (RecursiveCharacter...) │
                   └─────────────┬──────────┘
                                 │
                                 ▼
                   ┌────────────────────────┐
                   │   Mistral Embeddings    │
                   │     (mistral-embed)     │
                   └─────────────┬──────────┘
                                 │
                                 ▼
                     ┌────────────────────┐
                     │     ChromaDB       │
                     │ (Vector Database)  │
                     └─────────┬──────────┘
                               │
                 Query: "John" │
                               ▼
                   ┌────────────────────────┐
                   │  Semantic Retrieval     │
                   └─────────────┬──────────┘
                                 │
                                 ▼
                   ┌────────────────────────┐
                   │  LLM Augmented Prompt   │
                   │ (strict JSON schema)    │
                   └─────────────┬──────────┘
                                 │
                                 ▼
                   ┌────────────────────────┐
                   │   Mistral LLM Output    │
                   └────────────────────────┘
```

---

# 📁 Project Structure

```
langchain-character-extractor/
│── data/                       # Story files
│── chroma_db/                  # Vector DB (auto-created)
│── compute_embeddings.py       # Builds embeddings and DB
│── get_character_info.py       # RAG pipeline for character extraction
│── cli.py                      # User-friendly CLI
│── README.md
│── requirements.txt
│── .env.example
│── .gitignore
```

---

# ⚙️ Installation

```bash
git clone https://github.com/AlankarJagtap/langchain-character-extractor
cd langchain-character-extractor
pip install -r requirements.txt
cp .env.example .env
```

Add your API key:

```
MISTRAL_API_KEY=your_api_key_here
```

---

# 🚀 Usage

### 1️⃣ Compute Embeddings

```bash
python cli.py compute-embeddings --data-dir data --persist-dir chroma_db
```

### 2️⃣ Extract Character Information

```bash
python cli.py get-character-info "Alice"
```

---

# 🧪 Example Output

```json
{
  "name": "John Spatter",
  "storyTitle": "David Copperfield",
  "summary": "...",
  "relations": [
    {"name": "Michael", "relation": "business partner"}
  ],
  "characterType": "side character"
}
```

---

# 🛑 Edge Case Handling

### ❌ Character not found
```json
{ "error": "Character 'X' not found in any story." }
```

### ❌ Non-human term
```json
{ "error": "Not a character in the story." }
```

### ❌ Invalid JSON
Shows LLM output for debugging.

---

# 🎯 Summary

This project demonstrates:

- A **full RAG pipeline**
- Clean abstraction layers  
- Accurate retrieval via embeddings  
- Real-world structured LLM extraction  
- Robust error handling  
- Practical application of LangChain + MistralAI  

Perfect for interviews, assignments, and demonstrating knowledge of applied RAG systems.

---

📝 **Author:** Alankar Jagtap  
🔗 GitHub: https://github.com/AlankarJagtap
