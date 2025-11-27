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
4. **Augment LLM input** with retrieved chunks
5. **Generate structured JSON** using an LLM (`open-mistral-7b`)

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
│── data/                       
│── chroma_db/                  
│── compute_embeddings.py       
│── get_character_info.py       
│── cli.py                      
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

# 🎬 Demo Walkthrough

A full step-by-step demonstration on how to run the system.

---

## 🧰 1️⃣ Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your Mistral API key inside `.env`.

---

## 📚 2️⃣ Add Story Files

Place `.txt` stories inside:

```
data/
├── story1.txt
├── story2.txt
└── story3.txt
```

The **first line becomes the story title**.

---

## ⚙️ 3️⃣ Compute Embeddings

```bash
python cli.py compute-embeddings --data-dir data --persist-dir chroma_db
```

Expected output:

```
📘 Loading stories from: data
📄 Loaded X story files.
✂️ N chunks created.
💾 Saving embeddings into ChromaDB...
✅ Embeddings computed and stored successfully.
```

---

## 🔍 4️⃣ Extract Character Information

```bash
python cli.py get-character-info "John Spatter"
```

Example output:

```json
{
  "name": "John Spatter",
  "storyTitle": "The Poor Relation’s Story",
  "summary": "...",
  "relations": [
    {"name": "Michael", "relation": "friend and business partner"}
  ],
  "characterType": "side character"
}
```

---

## 🧪 5️⃣ Edge Case Demonstration

### ❌ Unknown character

```bash
python cli.py get-character-info "XYZPerson"
```

Result:

```json
{ "error": "Character 'XYZPerson' not found in any story." }
```

### ❌ Non-human entity

```bash
python cli.py get-character-info "School"
```

Result:

```json
{ "error": "Not a character in the story." }
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

# 🎯 Summary

This project demonstrates a complete, production-style **RAG pipeline**, combining:

- ChromaDB vector search  
- Mistral embeddings  
- LLM-based structured extraction  
- Strict hallucination prevention  
- Clean CLI workflows  

---

📝 **Author:** Alankar Jagtap  
🔗 **GitHub:** https://github.com/AlankarJagtap
