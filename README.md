## 🤖 My Chatbot
A multi-modal AI chatbot built with Streamlit and Groq, combining text chat, voice input/output, image understanding, and document-grounded Q&A (RAG) in a single app.
Features

💬 Text Chat — Fast streaming responses powered by Groq's LPU inference

🎙️ Voice Input — Record a message; speech is transcribed automatically using Whisper

🔊 Voice Output — Toggle on spoken replies using Groq's PlayAI text-to-speech

🖼️ Image Understanding — Attach an image and ask questions about it using a vision-capable model

📄 RAG (Retrieval-Augmented Generation) — Upload a PDF or TXT file and the bot answers questions grounded in that document, using local embeddings + FAISS similarity search

## Tech Stack
*Component/Technology

Frontend / UI- Streamlit

LLM (text)- [Groq — openai/gpt-oss-120b]

LLM (vision)-[Groq — qwen/qwen3.6-27b]

Speech-to-Text-[Groq — whisper-large-v3-turbo]

Text-to-Speech- [Groq — playai-tts]

Embeddings- sentence-transformers (all-MiniLM-L6-v2)

Vector Search- FAISS

PDF Parsing- PyPDF2

## Setup
1. Clone the repo
2. Create a virtual environment (Python 3.11 recommended)
3. Install dependencies
4. Add your Groq API key
5. Run the app
