import hashlib
import io
import math

import streamlit as st
from openai import OpenAI
from gliner import GLiNER

BASE_URL = "http://localhost:8033/v1"
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "text-embedding-3-small"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.2

# SYSTEM_PROMPT = (
#     "Tu es EcoBot, un assistant IA frugal conçu pour cet atelier d'IA locale. "
#     "Tu fonctionnes entièrement en local sur la machine pour limiter l'empreinte carbone. "
#     "\n\n"
#     "TES DIRECTIVES :"
#     "\n1. SOURCE OBLIGATOIRE : Tu ne dois répondre qu'en utilisant EXCLUSIVEMENT les informations "
#     "contenues dans le bloc 'CONTEXTE DOCUMENTAIRE' fourni."
#     "\n2. HONNÊTETÉ RADICALE : Si la réponse à la question n'est pas présente dans le contexte, "
#     "ne l'invente pas. Dis explicitement : 'Je ne trouve pas cette information dans les documents fournis (sobriété oblige).'"
#     "\n3. STYLE : Sois concis, direct et bienveillant. Réponds toujours en français."
# )

SYSTEM_PROMPT = (
    "Salut ! Je suis EcoBot 🌿, ton compagnon d'atelier pour une IA plus verte."
    "\nJe n'ai pas accès à internet, je ne connais que ce que tu me donnes à lire."
    "\n\n"
    "RÈGLES DU JEU :"
    "\n- Si l'info est dans tes documents : Je te la donne avec plaisir."
    "\n- Si l'info n'y est pas : Je te le dirai (je ne gaspille pas d'énergie à inventer des réponses)."
    "\n- Reste concis pour économiser des tokens !"
)

# ==============================================================================
# CLASSE GDPR SHIELD (GLiNER)
# ==============================================================================

@st.cache_resource
def load_pii_model():
    """Charge le modèle GLiNER depuis le dossier local."""
    # MODIFICATION ICI : On pointe vers le dossier local
    # Assurez-vous que le chemin est correct par rapport à votre script
    local_path = "./models/gliner_local" 
    
    try:
        # On ajoute load_tokenizer=True pour forcer l'usage des fichiers locaux
        return GLiNER.from_pretrained(local_path, load_tokenizer=True)
    except Exception as e:
        # Fallback pour le debug si le dossier n'est pas trouvé
        st.error(f"Erreur chargement GLiNER local : {e}")
        st.stop()
        
class GDPRShield:
    def __init__(self):
        self.model = load_pii_model()
        # Entités à détecter et masquer
        self.labels = [
            "person",           # Noms
            "email",            # Emails
            "phone number",     # Téléphones
            "credit card",      # CB
            "social security number", # Sécu
            "password"          # Mots de passe
        ]

    def anonymize(self, text):
        """Détecte les entités sensibles et les remplace par des balises <TAG_MASKED>."""
        if not text.strip():
            return text, 0
            
        # Prédiction (threshold=0.3 est un bon équilibre)
        entities = self.model.predict_entities(text, self.labels, threshold=0.3)
        
        if not entities:
            return text, 0
            
        # Tri inverse pour remplacer sans casser les index
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        text_list = list(text)
        count = 0
        
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['label']
            
            replacement = f"<{label.upper().replace(' ', '_')}_MASKED>"
            text_list[start:end] = list(replacement)
            count += 1
            
        return "".join(text_list), count


# ==============================================================================
# FONCTIONS UTILITAIRES (RAG)
# ==============================================================================
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


# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="EcoBot Local", page_icon="🌿")
st.title("🌿 EcoBot : Le Chatbot Vert")

# Initialisation du client OpenAI (compatible llama.cpp)
client = OpenAI(
    base_url=BASE_URL,
    api_key="sk-no-key-needed",
)

# Initialisation du Shield (lazy loading via cache)
gdpr_shield = GDPRShield()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT

if "rag_docs" not in st.session_state:
    reset_rag_state()

# --- SIDEBAR ---
st.sidebar.image("logo.png", width=150, )
st.sidebar.title("Configuration")
# SECTION PRIVACY
st.sidebar.subheader("🛡️ Privacy Shield")
use_gdpr_shield = st.sidebar.checkbox(
    "Activer Anonymisation (GLiNER)", 
    value=False,
    help="Utilise un petit modèle IA pour masquer les données personnelles (Noms, Emails...) dans les fichiers ET le chat."
)

st.sidebar.divider()
st.sidebar.title("Documents (RAG)")

# CSS pour les infobulles
st.sidebar.markdown(
    """
    <style>
    .topk-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        margin: 0.1rem 0 0.2rem 0;
    }
    .topk-label .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.1rem;
        height: 1.1rem;
        border: 1px solid #999;
        border-radius: 999px;
        font-size: 0.75rem;
        line-height: 1rem;
        color: #666;
        background: #f3f3f3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar.expander("Prompt systeme", expanded=False):
    st.text_area(
        "Visible par l'utilisateur et modifiable",
        key="system_prompt",
        height=200,
    )
    if st.button("Reinitialiser prompt systeme"):
        st.session_state.system_prompt = SYSTEM_PROMPT

with st.sidebar.expander("Ajouter des documents", expanded=False):
    uploaded_files = st.file_uploader(
        "Ajouter un ou plusieurs fichiers (txt, md, pdf)",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    st.markdown(
        """
        <div class="topk-label" title="Longueur des morceaux qu'on decoupe dans vos documents. Plus grand = plus long a lire, plus petit = plus de morceaux.">
            <span class="info-icon">i</span>
            <span>Taille chunk</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chunk_size = st.number_input(
        "Taille chunk (caracteres)",
        min_value=200,
        max_value=4000,
        value=DEFAULT_CHUNK_SIZE,
        step=100,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="topk-label" title="La partie qui se repete entre deux morceaux. Plus grand = plus de recouvrement pour ne rien rater.">
            <span class="info-icon">i</span>
            <span>Recouvrement</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chunk_overlap = st.number_input(
        "Recouvrement (caracteres)",
        min_value=0,
        max_value=1000,
        value=DEFAULT_CHUNK_OVERLAP,
        step=50,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="topk-label" title="Combien d'extraits de vos documents on garde pour repondre. Plus grand = plus d'infos, mais parfois moins precis.">
            <span class="info-icon">i</span>
            <span>Top K</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=8,
        value=DEFAULT_TOP_K,
        step=1,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="topk-label" title="Le niveau minimum de ressemblance pour garder un extrait. Plus haut = plus strict, plus bas = plus large.">
            <span class="info-icon">i</span>
            <span>Seuil de similarite</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    min_score = st.slider(
        "Seuil similarite",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MIN_SCORE,
        step=0.05,
        label_visibility="collapsed",
    )

    if st.button("Effacer documents"):
        reset_rag_state()

    st.caption("Note: pour les embeddings, lancez llama.cpp avec --embedding.")

# TRAITEMENT DES FICHIERS UPLOADÉS
if uploaded_files:
    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        file_hash = hashlib.sha256(data).hexdigest()

        # Vérification doublon basé sur le hash du fichier brut
        if file_hash in st.session_state.rag_hashes:
            continue

        text, error = text_from_bytes(uploaded_file.name, data)
        if error:
            st.sidebar.error(f"{uploaded_file.name}: {error}")
            continue
        
        # --- LOGIQUE GDPR SHIELD (DOCUMENTS) ---
        if use_gdpr_shield:
            with st.spinner(f"🛡️ Anonymisation IA en cours : {uploaded_file.name}"):
                text, count_pii = gdpr_shield.anonymize(text)
            if count_pii > 0:
                st.sidebar.warning(f"🛡️ {count_pii} éléments masqués dans {uploaded_file.name}")
        # ---------------------------------------

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

# --- ZONE DE CHAT ---

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # 1. On affiche le message de l'utilisateur tel quel dans l'interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. PRÉPARATION DU PROMPT SÉCURISÉ
    safe_prompt = prompt
    if use_gdpr_shield:
        with st.spinner("Anonymisation de votre question..."):
            safe_prompt, count_prompt_pii = gdpr_shield.anonymize(prompt)
        
        if count_prompt_pii > 0:
            st.info(f"🔒 Votre prompt a été sécurisé avant envoi : \"{safe_prompt}\"")

    # 3. STOCKAGE DE LA VERSION SÉCURISÉE
    # On stocke la version 'safe' dans l'historique pour que le LLM ne voit jamais les infos persos
    st.session_state.messages.append({"role": "user", "content": safe_prompt})

    # 4. RECHERCHE RAG (Sur le prompt sécurisé !)
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

    # 5. On prépare le contenu du message final (System + Contexte + Question)
    # On commence par l'instruction système
    final_content = st.session_state.system_prompt + "\n\n"

    # On ajoute le contexte s'il existe
    if retrieved_chunks:
        context_text = "\n\n".join(
            f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(retrieved_chunks)
        )
        final_content += f"CONTEXTE DOCUMENTAIRE :\n{context_text}\n\n"

    # On ajoute la question de l'utilisateur (le dernier message ajouté dans la session)
    last_user_message = st.session_state.messages[-1]["content"]
    final_content += f"QUESTION UTILISATEUR :\n{last_user_message}"

    # 6. On construit la liste des messages pour le LLM
    # On prend tout l'historique SAUF le dernier message (qu'on va remplacer par notre version augmentée)
    messages_for_llm = st.session_state.messages[:-1]
    
    # On ajoute notre gros message fusionné à la fin
    messages_for_llm.append({"role": "user", "content": final_content})

    # 7. APPEL LLM EN STREAMING
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
