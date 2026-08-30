<div align="center">

# ✦ Sidera

### Local, private, document-grounded AI.

A fully local **Retrieval-Augmented Generation (RAG)** assistant for chatting with PDF documents using **Microsoft Foundry Local**.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Vector%20Storage-003B57?logo=sqlite&logoColor=white)
![Local RAG](https://img.shields.io/badge/RAG-100%25%20Local-7C3AED)
![Foundry Local](https://img.shields.io/badge/Microsoft-Foundry%20Local-5E5CE6)

</div>

<p align="center">
  <img src="assets/Ekran%20Resmi%202026-08-31%2000.11.33.png" alt="Sidera local RAG interface" width="100%">
</p>

---

## Overview

**Sidera** lets users upload one or more PDFs, index them locally, and ask questions through a Streamlit chat interface.

The complete RAG pipeline runs on-device: document parsing, chunking, embeddings, retrieval, and language-model inference. Answers are grounded in the indexed document collection, and unsupported questions are rejected instead of answered from general knowledge.

> **No cloud-hosted LLM API is required for document Q&A.**

---

## Interface Preview

Sidera guides the user through the complete local RAG workflow directly in the interface: starting the local models, uploading PDFs, indexing their chunks, and chatting with grounded answers.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.11.02.png" alt="Sidera local model startup screen" width="100%"><br>
      <sub><b>1. Local startup</b> — prepares the knowledge engine and on-device models.</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.11.14.png" alt="Sidera PDF upload screen" width="100%"><br>
      <sub><b>2. PDF upload</b> — starts the local document-processing pipeline.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.10.13.png" alt="Sidera document indexing progress" width="100%"><br>
      <sub><b>3. Local indexing</b> — chunks and embeds the document locally.</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.10.30.png" alt="Sidera grounded document chat" width="100%"><br>
      <sub><b>4. Grounded chat</b> — answers from retrieved document context and rejects unsupported questions.</sub>
    </td>
  </tr>
</table>

---

## Highlights

| Capability | Implementation |
|---|---|
| 📄 **Multi-PDF Upload** | Upload and index multiple PDFs in the same local knowledge base |
| 🧩 **Chunking** | Extract and split document text into retrievable chunks |
| 🧠 **Local Embeddings** | `qwen3-embedding-0.6b` |
| 🔎 **Semantic Retrieval** | Cosine similarity over locally stored vectors |
| 💾 **Local Storage** | SQLite-backed document and embedding storage |
| 🤖 **Local Chat Model** | `phi-3.5-mini` through Microsoft Foundry Local |
| 🛡️ **Grounded Answers** | Answers use only retrieved document context |
| 🚫 **Out-of-Context Rejection** | Unsupported questions return a safe fallback |
| 📚 **Source Inspection** | View retrieved chunks and similarity scores |
| 📝 **Extractive Summaries** | Document summaries prioritize source-faithful sentences |
| 🌓 **Interface** | Dark and light appearance modes |

---

## How It Works

```text
┌──────────────┐
│  PDF Upload  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Text Extract │
└──────┬───────┘
       ↓
┌──────────────┐
│   Chunking   │
└──────┬───────┘
       ↓
┌────────────────────┐
│ qwen3 Embeddings   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ SQLite Vector Data │
└─────────┬──────────┘
          │
          │ User Question
          ↓
┌────────────────────┐
│  Query Embedding   │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Cosine Similarity  │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Relevant Chunks    │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│   phi-3.5-mini     │
│  Foundry Local     │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│ Grounded Response  │
└────────────────────┘
```

---

## Grounded Q&A

Sidera retrieves the most relevant document chunks before generating an answer.

For normal questions:

```text
Question
   ↓
Query Embedding
   ↓
Top Relevant Chunks
   ↓
Local LLM
   ↓
Grounded Answer
```

If the indexed document does not provide enough evidence, Sidera responds with:

```text
I don't know based on the provided documents.
```

### Example

**Question**

```text
What happens during a power outage longer than 15 minutes?
```

**Sidera**

```text
During a power outage longer than 15 minutes, the Lead Baker
records the refrigerator temperature. If refrigerated ingredients
rise above 8°C, those ingredients are isolated and not used until
a safety decision is made.
```

An unrelated question such as:

```text
What is the capital of Brazil?
```

returns:

```text
I don't know based on the provided documents.
```

---

## Source Inspection

Answers expose the retrieval layer instead of hiding it.

```text
Sources · 3 retrieved chunks
```

Each retrieved source can include:

- document name
- chunk index
- cosine similarity score

This makes retrieval behavior easier to inspect, test, and debug.

---

## Multi-Document Retrieval & Performance

Sidera can keep multiple PDFs in the same local knowledge base and retrieve the most relevant chunks across documents. The source panel shows which PDF, chunk, and similarity score contributed to an answer.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.42.05.png" alt="Sidera multi-document retrieval using the Summer School project plan" width="100%"><br>
      <sub><b>Cross-document Q&A</b> — retrieves the project-plan phases from the Summer School PDF.</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/Ekran%20Resmi%202026-08-31%2000.41.56.png" alt="Sidera source inspection for Moonlight Bakery PDF" width="100%"><br>
      <sub><b>Source-aware retrieval</b> — answers from the Moonlight Bakery PDF and exposes source chunks with similarity scores.</sub>
    </td>
  </tr>
</table>

### Performance

<p align="center">
  <img src="assets/Ekran%20Resmi%202026-08-31%2000.41.47.png" alt="Sidera terminal performance metrics" width="92%">
</p>

<p align="center">
  <sub><b>Local performance timing</b> — retrieval, generation, and total response time are measured in the terminal for each query.</sub>
</p>

---

## Extractive Document Summaries

Broad requests such as:

```text
Summarize this document in 5 key points.
```

use a dedicated extractive summarization flow.

```text
Retrieved Chunks
      ↓
Sentence Extraction
      ↓
Fragment Filtering
      ↓
Near-Duplicate Removal
      ↓
Representative Sentence Selection
      ↓
Extractive Summary
```

The final summary uses selected document sentences instead of freely rewriting multiple unrelated chunks. This reduces unsupported combinations and preserves source grounding.

---

## Models

### Embedding

```text
qwen3-embedding-0.6b
```

Used for both document chunks and user queries so semantic similarity can be calculated in the same vector space.

### Chat

```text
phi-3.5-mini
```

Runs locally through **Microsoft Foundry Local** and generates answers from retrieved document context.

---

## Tech Stack

```text
Python
├── Streamlit
├── Microsoft Foundry Local
├── PyMuPDF
├── SQLite
├── qwen3-embedding-0.6b
└── phi-3.5-mini
```

---

## Project Structure

```text
sidera-local-rag/
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── Ekran Resmi 2026-08-31 00.10.13.png
│   ├── Ekran Resmi 2026-08-31 00.10.30.png
│   ├── Ekran Resmi 2026-08-31 00.11.02.png
│   ├── Ekran Resmi 2026-08-31 00.11.14.png
│   ├── Ekran Resmi 2026-08-31 00.11.33.png
│   ├── Ekran Resmi 2026-08-31 00.41.47.png
│   ├── Ekran Resmi 2026-08-31 00.41.56.png
│   └── Ekran Resmi 2026-08-31 00.42.05.png
│
├── data/
├── documents/
├── models/
│
└── src/
    ├── __init__.py
    ├── chat.py
    ├── config.py
    ├── database.py
    ├── embeddings.py
    ├── evaluate.py
    ├── ingest.py
    ├── retrieval.py
    └── utils.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ezgibaylamm/sidera-local-rag.git
cd sidera-local-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

Start Sidera with:

```bash
streamlit run app.py
```

Then open the local Streamlit address displayed in the terminal, upload one or more PDFs, wait for indexing to complete, and start asking questions.

---

## Example Prompts

```text
What is this document mainly about?
```

```text
What are the main topics discussed in this document?
```

```text
Summarize this document in 5 key points.
```

```text
What are the four tracked allergens?
```

```text
What happens during a power outage longer than 15 minutes?
```

---

## Design Goals

Sidera was built around four priorities:

**Local-first**  
Document processing and model inference stay on the local machine.

**Grounded**  
Answers are generated from retrieved document evidence rather than unrestricted model knowledge.

**Inspectable**  
Retrieved chunks and similarity scores can be viewed directly in the interface.

**Practical**  
The project covers the complete RAG lifecycle, from PDF upload to an interactive local chat experience.

---

## Current Status

- [x] Web-based multi-PDF upload
- [x] Multi-document local knowledge base
- [x] Local PDF ingestion
- [x] Text chunking
- [x] Local embeddings
- [x] SQLite storage
- [x] Cosine-similarity retrieval
- [x] Grounded Q&A
- [x] Source inspection
- [x] Out-of-context rejection
- [x] Extractive document summarization
- [x] Streamlit chat interface
- [x] Dark / light appearance
- [x] Microsoft Foundry Local inference
- [x] Retrieval / generation / total response-time logging

---

<div align="center">

### ✦ Sidera

**Local Knowledge Intelligence**

Built by **Ezgi Baylam**

</div>
