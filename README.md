✦ Sidera — Local RAG Assistant

Sidera is a fully local Retrieval-Augmented Generation (RAG) application for asking questions about PDF documents.

The application allows users to upload a PDF through a Streamlit interface, processes and indexes the document locally, retrieves relevant document chunks using semantic similarity, and generates grounded answers with a locally running language model through Microsoft Foundry Local.

No cloud-based language model API is required for document question answering.

✨ Features

PDF upload directly from the web interface

Local PDF text extraction

Automatic document chunking

Local embedding generation

SQLite-based document and vector storage

Cosine similarity retrieval

Grounded document question answering

Out-of-context question rejection

Retrieved source and chunk inspection

Extractive document summarization

Dark and light interface modes

Fully local model inference with Microsoft Foundry Local

🧠 Models

Embedding Model

qwen3-embedding-0.6b

Used to convert document chunks and user queries into vector representations for semantic retrieval.

Chat Model

phi-3.5-mini

Used locally to generate answers from retrieved document context.

Both models are managed through Microsoft Foundry Local.

🏗️ Architecture

Sidera follows a local RAG pipeline:

PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Generation
    ↓
SQLite Storage
    ↓
User Question
    ↓
Query Embedding
    ↓
Cosine Similarity Search
    ↓
Relevant Chunks
    ↓
Local LLM
    ↓
Grounded Answer

For document summaries, Sidera uses a separate extractive summarization flow:

Retrieved Document Chunks
    ↓
Sentence Extraction
    ↓
Fragment Filtering
    ↓
Duplicate Removal
    ↓
Sentence Selection
    ↓
Extractive Summary

This approach reduces the risk of combining unrelated facts or introducing unsupported information during summarization.

🔍 Retrieval

When a user asks a question:

The question is embedded using the same embedding model used for the document.

Cosine similarity is calculated between the query vector and stored document vectors.

The most relevant chunks are retrieved.

Retrieved chunks are passed to the local language model as context.

The model generates an answer grounded in the retrieved document information.

For normal Q&A, Sidera retrieves the most relevant chunks.

For broader document-summary requests, a larger set of chunks is used to provide wider document coverage.

🛡️ Grounded Responses

Sidera is designed to answer questions using only the indexed document context.

If the retrieved context does not contain enough information, the assistant responds with:

I don't know based on the provided documents.

For example:

User:
What is the capital of Brazil?

Sidera:
I don't know based on the provided documents.

This behavior helps reduce unsupported answers and hallucinations.

📚 Source Inspection

Each generated answer can display the document chunks used during retrieval.

Example:

Sources · 3 retrieved chunks

For each source, Sidera can show information such as:

Source document

Chunk number

Similarity score

This makes the retrieval process easier to inspect and debug.

📄 PDF Upload

Documents can be uploaded directly through the Streamlit interface.

After a PDF is uploaded, Sidera:

Extracts the document text

Splits the text into chunks

Generates embeddings

Stores the chunks and vectors in SQLite

Makes the document available for local question answering

The user can start chatting with the document after indexing is completed.

📝 Extractive Summarization

Document-wide summary requests use an extractive summarization pipeline.

Instead of allowing the language model to freely rewrite multiple document sections, Sidera:

extracts complete sentences from retrieved chunks,

removes incomplete chunk fragments,

removes near-duplicate sentences caused by chunk overlap,

filters context-dependent fragments,

asks the local model to select representative sentences,

returns the original selected document sentences without rewriting them.

Example:

Summarize this document in 5 key points.

This design prioritizes factual grounding over generative rewriting.

🗂️ Project Structure

sidera-local-rag/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   └── rag.db
│
├── documents/
│
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

⚙️ Technologies

Python

Streamlit

Microsoft Foundry Local

SQLite

PyMuPDF

Qwen3 Embedding

Phi-3.5 Mini

Cosine Similarity

Retrieval-Augmented Generation (RAG)

🚀 Installation

Clone the repository:

git clone https://github.com/ezgibaylamm/sidera-local-rag.git

Enter the project directory:

cd sidera-local-rag

Create a virtual environment:

python3 -m venv .venv

Activate it on macOS/Linux:

source .venv/bin/activate

Install the dependencies:

pip install -r requirements.txt

▶️ Running Sidera

Start the Streamlit application:

streamlit run app.py

Then open the local address displayed by Streamlit in your browser.

Usually:

http://localhost:8501

Upload a PDF, wait for local indexing to complete, and start asking questions.

💬 Example Questions

What is this document mainly about?

What are the main topics discussed in this document?

Summarize this document in 5 key points.

What happens during a power outage longer than 15 minutes?

What are the four tracked allergens?

🔒 Local Processing

Sidera is designed around local document processing.

The document pipeline, embeddings, vector retrieval, and language model inference run locally through the application and Microsoft Foundry Local.

This architecture is useful for:

private documents,

offline experimentation,

local AI development,

RAG prototyping,

educational projects,

environments where sending document content to an external LLM API is undesirable.

🎯 Project Goals

The project was built to explore and implement the complete lifecycle of a local RAG system:

document ingestion,

text chunking,

embeddings,

vector similarity,

retrieval,

grounded prompting,

local inference,

hallucination reduction,

source inspection,

document summarization,

and interactive UI design.

The main goal is to demonstrate how a practical document-grounded AI assistant can operate locally without relying on a cloud-hosted LLM API.

📌 Current Status

Sidera currently supports:

✅ PDF upload

✅ Local document indexing

✅ Semantic retrieval

✅ SQLite vector storage

✅ Grounded Q&A

✅ Source inspection

✅ Out-of-context rejection

✅ Extractive document summaries

✅ Streamlit web interface

✅ Dark / light appearance

✅ Local Foundry model inference

👩‍💻 Author

Ezgi Baylam

Software Engineering