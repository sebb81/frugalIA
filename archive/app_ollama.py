import os
import streamlit as st
import numpy as np
import ollama  # Importation de la bibliothèque Ollama

# --- Configuration ---
# Spécifiez le nom du modèle Ollama que vous souhaitez utiliser
MODEL_NAME = "mistral"  # Assurez-vous d'avoir fait 'ollama pull mistral'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialisation des modèles
if "embedder" not in st.session_state:
    from sentence_transformers import SentenceTransformer
    st.session_state.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Note : Avec Ollama, on n'a pas besoin de charger le LLM en mémoire dans session_state 
# comme avec llama-cpp, car Ollama gère le cycle de vie du modèle en arrière-plan.

# Initialisation de la base de connaissances
if "docs" not in st.session_state:
    st.session_state.docs = []        
    st.session_state.doc_names = []   
    st.session_state.embeddings = []  

# --- Barre latérale (options) ---
st.sidebar.title("Documents & Options")
uploaded_file = st.sidebar.file_uploader("Ajouter un document (txt ou pdf)", type=["txt", "pdf"])

if uploaded_file:
    file_name = uploaded_file.name
    if file_name.endswith(".pdf"):
        try:
            import fitz 
        except ImportError:
            os.system("pip install pymupdf")
            import fitz
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Découpage simple
    chunks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph: continue
        words = paragraph.split()
        if len(words) > 200:
            for i in range(0, len(words), 200):
                chunks.append(" ".join(words[i:i+200]))
        else:
            chunks.append(paragraph)

    # Calcul des embeddings
    new_embeddings = st.session_state.embedder.encode(chunks, normalize_embeddings=True)
    st.session_state.docs.extend(chunks)
    st.session_state.doc_names.extend([file_name] * len(chunks))
    st.session_state.embeddings.extend(new_embeddings)
    st.sidebar.write(f"✅ Document '{file_name}' indexé.")

# Affichage des documents indexés
if st.session_state.doc_names:
    st.sidebar.markdown("**Documents indexés :**")
    for name in set(st.session_state.doc_names):
        st.sidebar.write(f"- *{name}*")

show_stats = st.sidebar.checkbox("Afficher les stats de génération (Ollama)")

# --- Zone de question/réponse ---
st.title("💬 Chatbot anti-fraude (Ollama)")
st.write("Posez une question à partir des documents fournis.")

user_question = st.text_input("Votre question :", "")

if user_question:
    if not st.session_state.docs:
        st.warning("⚠️ Veuillez d'abord ajouter des documents.")
    else:
        # 1. Recherche de similarité
        query_emb = st.session_state.embedder.encode(user_question, normalize_embeddings=True)
        doc_embeddings = np.array(st.session_state.embeddings)
        sims = np.dot(doc_embeddings, query_emb)
        top_idx = sims.argsort()[::-1]
        
        top_k = 2
        threshold = 0.3
        retrieved_chunks = []
        used_sources = set()
        
        for idx in top_idx[:top_k]:
            if sims[idx] >= threshold:
                retrieved_chunks.append(st.session_state.docs[idx])
                used_sources.add(st.session_state.doc_names[idx])

        if not retrieved_chunks:
            answer = "Désolé, je ne trouve pas d'information pertinente dans les documents."
        else:
            context_text = "\n".join([f"- {c}" for c in retrieved_chunks])
            
            system_prompt = (
                "Tu es un assistant IA expert en fraude, et tu réponds exclusivement en français. "
                "Tu réponds uniquement à partir des informations fournies dans le contexte. "
                "Si la réponse n'est pas dans le contexte, dis-le. "
                "Sois court, clair et professionnel."
            )

            # 2. Appel de l'API Ollama
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': f"CONTEXTE:\n{context_text}\n\nQUESTION: {user_question}"},
                    ],
                    options={
                        'temperature': 0.2, # Plus précis
                        'num_predict': 256,   # Équivalent max_tokens
                    }
                )
                answer = response['message']['content']
                
                # Affichage de la réponse
                st.success(answer)
                if used_sources:
                    st.write("**Sources :** " + ", ".join(used_sources))
                
                # Stats (tokens)
                if show_stats:
                    # Ollama fournit directement le compte de tokens
                    eval_count = response.get('eval_count', 0)
                    st.caption(f"*Réponse générée en {eval_count} tokens*")
            
            except Exception as e:
                st.error(f"Erreur de connexion à Ollama : {e}")
                st.info("Vérifiez que le serveur Ollama est bien lancé (ollama serve)")