import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="My Chatbot", page_icon="🤖")
st.title("🤖 My Chatbot")

TEXT_MODEL = "openai/gpt-oss-120b"        # current Groq text model (replaces deprecated llama-3.3-70b-versatile)
VISION_MODEL = "qwen/qwen3.6-27b"         # current Groq vision model
STT_MODEL = "whisper-large-v3-turbo"      
TTS_MODEL = "playai-tts"                  # text-to-speech
TTS_VOICE = "Fritz-PlayAI"                # pick any supported PlayAI voice


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")  


embedder = load_embedder()


def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8")


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def build_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve(query, chunks, index, k=3):
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def text_to_speech(text):
    """Returns raw audio bytes for the given text using Groq's PlayAI TTS."""
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    return response.read()


with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    voice_reply = st.checkbox("Speak the bot's replies", value=False)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pop("rag_index", None)
        st.session_state.pop("rag_chunks", None)
        st.session_state.pop("last_doc", None)
        st.rerun()

    st.divider()
    st.subheader("📄 RAG: Chat with a document")
    doc = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], key="rag_doc")

    if doc is not None and st.session_state.get("last_doc") != doc.name:
        with st.spinner("Reading and indexing document..."):
            text = extract_text(doc)
            chunks = chunk_text(text)
            st.session_state.rag_chunks = chunks
            st.session_state.rag_index = build_index(chunks)
            st.session_state.last_doc = doc.name
        st.success(f"Indexed {len(chunks)} chunks from {doc.name}")

    use_rag = st.checkbox("Answer using document", value="rag_index" in st.session_state)



if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, friendly assistant."}
    ]


for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):  # image message
            for part in msg["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    st.image(part["image_url"]["url"])
        else:
            st.markdown(msg["content"])



uploaded_image = st.file_uploader("Attach an image (optional)", type=["png", "jpg", "jpeg"], key="chat_image")
audio_value = st.audio_input(
    "Or record a voice message",
    key="audio_input"
)

user_text = None

if audio_value is not None:
    transcript = client.audio.transcriptions.create(
        file=("audio.wav", audio_value.read()),
        model=STT_MODEL,
    )
    user_text = transcript.text
    st.info(f"🎙️ Transcribed: {user_text}")

typed_text = st.chat_input("Ask me anything...")


if audio_value is not None:
    transcript = client.audio.transcriptions.create(
        file=("audio.wav", audio_value.read()),
        model=STT_MODEL,
    )
    user_text = transcript.text
    st.info(f"Transcribed: {user_text}")

if typed_text:
    user_text = typed_text


if user_text:

    
    if use_rag and "rag_index" in st.session_state:
        relevant_chunks = retrieve(user_text, st.session_state.rag_chunks, st.session_state.rag_index)
        context = "\n\n".join(relevant_chunks)
        augmented_text = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {user_text}"""
    else:
        augmented_text = user_text

    
    if uploaded_image is not None:
        b64_image = base64.b64encode(uploaded_image.read()).decode("utf-8")
        mime = uploaded_image.type
        content = [
            {"type": "text", "text": augmented_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
        ]
        model_to_use = VISION_MODEL
    else:
        content = augmented_text
        model_to_use = TEXT_MODEL

    st.session_state.messages.append({"role": "user", "content": content})

    with st.chat_message("user"):
        # Show what the user actually asked, not the RAG-stuffed prompt
        st.markdown(user_text)
        if uploaded_image is not None:
            st.image(uploaded_image)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

    if voice_reply and full_response.strip():
        with st.spinner("Generating voice reply..."):
            audio_bytes = text_to_speech(full_response)
        st.audio(audio_bytes, format="audio/wav")

        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})