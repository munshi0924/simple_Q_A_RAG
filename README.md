# simple_Q_A_RAG

simple_Q_A_RAG is an interactive Retrieval-Augmented Generation (RAG) system built with LangChain, OpenAI, ChromaDB, and Gradio. It allows users to upload any PDF document and ask grounded questions about its contents.

---

## Features

- **PDF Ingestion & Processing:** Parses documents and splits text using recursive character chunking (1,000 chars, 200 overlap).
- **Vector Search:** Converts chunks into vector embeddings (`text-embedding-3-small`) and indexes them locally with ChromaDB.
- **Context Grounding:** Restricts LLM answers strictly to the retrieved top-3 document passages to prevent hallucinations.
- **Interactive Chat Interface:** Provides a lightweight Web UI powered by Gradio.

---

## Tech Stack

- **UI Framework:** Gradio
- **Orchestration:** LangChain
- **LLM:** OpenAI `gpt-3.5-turbo`
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Database:** ChromaDB

---

## Project Architecture

1. **Document Loading:** `PyPDFLoader` extracts text pages from uploaded PDFs.
2. **Text Chunking:** `RecursiveCharacterTextSplitter` cuts documents into manageable pieces.
3. **Embedding Generation:** OpenAI Embeddings model converts text into dense vector representations.
4. **Retrieval:** ChromaDB queries the vector store to fetch the $k=3$ most relevant chunks.
5. **Generation:** `ChatOpenAI` synthesizes an answer using only the retrieved context.

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/munshi0924/simple_Q_A_RAG.git](https://github.com/munshi0924/simple_Q_A_RAG.git)
   cd simple_Q_A_RAG
