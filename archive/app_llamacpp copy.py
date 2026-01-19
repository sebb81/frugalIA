import hashlib
import io
import math

import streamlit as st
from openai import OpenAI

BASE_URL = "http://localhost:8033/v1"
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.2

SYSTEM_PROMPT = (
    "Tu es un assistant utile. Reponds en francais. "
    "Si un CONTEXTE est fourni, utilise-le en priorite. "
    "Si la reponse n'y est pas, dis-le clairement."
)

def text_from_bytes(name, data):
    name_lower = name.lower()
    if name_lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except Exception:
            return "", "PDF support requires pypdf (pip install pypdf)."
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages), None
    return data.decode("utf-8", errors="ignore"), None


def chunk_text(text, max_chars, overlap):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    effective_overlap = min(overlap, max_chars - 1) if max_chars > 1 else 0
    step = max(1, max_chars - effective_overlap)

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue

        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            for i in range(0, len(paragraph), step):
                chunks.append(paragraph[i : i + max_chars])
            current = ""

    if current:
        chunks.append(current)

    return chunks


def embed_texts(texts, client):
    embeddings = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


def dot_product(left, right):
    return sum(l * r for l, r in zip(left, right))


def vector_norm(vector):
    return math.sqrt(sum(v * v for v in vector))


def retrieve_chunks(query, top_k, min_score, client):
    query_embedding = embed_texts([query], client)[0]
    query_norm = vector_norm(query_embedding)
    if query_norm == 0:
        return [], []

    scored = []
    for idx, embedding in enumerate(st.session_state.rag_embeddings):
        denom = query_norm * st.session_state.rag_norms[idx]
        score = dot_product(query_embedding, embedding) / denom if denom else 0.0
        scored.append((score, idx))

    scored.sort(key=lambda item: item[0], reverse=True)
    chunks = []
    sources = set()
    for score, idx in scored[:top_k]:
        if score < min_score:
            continue
        chunks.append(st.session_state.rag_docs[idx])
        sources.add(st.session_state.rag_sources[idx])

    return chunks, sorted(sources)


def reset_rag_state():
    st.session_state.rag_docs = []
    st.session_state.rag_sources = []
    st.session_state.rag_embeddings = []
    st.session_state.rag_norms = []
    st.session_state.rag_hashes = set()


st.set_page_config(page_title="EcoBot Local", page_icon="🌿")
st.title("🌿 EcoBot : Le Chatbot Vert")

client = OpenAI(
    base_url=BASE_URL,
    api_key="sk-no-key-needed",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_docs" not in st.session_state:
    reset_rag_state()

st.sidebar.image("logo.png", width=360, )
st.sidebar.title("Documents (RAG)")

uploaded_files = st.sidebar.file_uploader(
    "Ajouter un ou plusieurs fichiers (txt, md, pdf)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)

chunk_size = st.sidebar.number_input(
    "Taille chunk (caracteres)",
    min_value=200,
    max_value=4000,
    value=DEFAULT_CHUNK_SIZE,
    step=100,
)
chunk_overlap = st.sidebar.number_input(
    "Recouvrement (caracteres)",
    min_value=0,
    max_value=1000,
    value=DEFAULT_CHUNK_OVERLAP,
    step=50,
)
top_k = st.sidebar.slider(
    "Top k",
    min_value=1,
    max_value=8,
    value=DEFAULT_TOP_K,
    step=1,
)
min_score = st.sidebar.slider(
    "Seuil similarite",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_MIN_SCORE,
    step=0.05,
)

if st.sidebar.button("Effacer documents"):
    reset_rag_state()

st.sidebar.caption("Note: pour les embeddings, lancez llama.cpp avec --embedding.")

if uploaded_files:
    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()
        if file_hash in st.session_state.rag_hashes:
            continue

        text, error = text_from_bytes(uploaded_file.name, data)
        if error:
            st.sidebar.error(f"{uploaded_file.name}: {error}")
            continue

        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            st.sidebar.warning(f"{uploaded_file.name}: fichier vide.")
            continue

        with st.spinner(f"Indexation: {uploaded_file.name}"):
            try:
                embeddings = embed_texts(chunks, client)
            except Exception as exc:
                st.sidebar.error(
                    "Embedding error. Verifiez llama.cpp (--embedding) et le modele charge."
                )
                st.sidebar.error(str(exc))
                continue

        st.session_state.rag_docs.extend(chunks)
        st.session_state.rag_sources.extend([uploaded_file.name] * len(chunks))
        st.session_state.rag_embeddings.extend(embeddings)
        st.session_state.rag_norms.extend([vector_norm(e) for e in embeddings])
        st.session_state.rag_hashes.add(file_hash)
        st.sidebar.success(
            f"Indexe: {uploaded_file.name} ({len(chunks)} chunks)"
        )

if st.session_state.rag_sources:
    st.sidebar.markdown("Documents indexes:")
    for name in sorted(set(st.session_state.rag_sources)):
        st.sidebar.write(f"- {name}")
    st.sidebar.caption(f"{len(st.session_state.rag_docs)} chunks indexes")
else:
    st.sidebar.info("Aucun document indexe.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved_chunks = []
    used_sources = []
    if st.session_state.rag_embeddings:
        with st.spinner("Recherche dans les documents..."):
            try:
                retrieved_chunks, used_sources = retrieve_chunks(
                    prompt, top_k, min_score, client
                )
            except Exception as exc:
                st.warning(f"RAG indisponible: {exc}")

    # 1. On prépare le contenu du message final (System + Contexte + Question)
    # On commence par l'instruction système
    final_content = SYSTEM_PROMPT + "\n\n"

    # On ajoute le contexte s'il existe
    if retrieved_chunks:
        context_text = "\n\n".join(
            f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(retrieved_chunks)
        )
        final_content += f"CONTEXTE DOCUMENTAIRE :\n{context_text}\n\n"

    # On ajoute la question de l'utilisateur (le dernier message ajouté dans la session)
    last_user_message = st.session_state.messages[-1]["content"]
    final_content += f"QUESTION UTILISATEUR :\n{last_user_message}"

    # 2. On construit la liste des messages pour le LLM
    # On prend tout l'historique SAUF le dernier message (qu'on va remplacer par notre version augmentée)
    messages_for_llm = st.session_state.messages[:-1]
    
    # On ajoute notre gros message fusionné à la fin
    messages_for_llm.append({"role": "user", "content": final_content})

    

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_for_llm,
            stream=True,
        )
        response = st.write_stream(stream)
        if used_sources:
            st.caption("Sources: " + ", ".join(used_sources))

    st.session_state.messages.append({"role": "assistant", "content": response})
