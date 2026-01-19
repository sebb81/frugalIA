import hashlib
import io
import math
import os
import pickle
import streamlit as st
from openai import OpenAI
from gliner import GLiNER
import logging

BASE_URL = "http://localhost:8033/v1"
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_FILE = "eco_index.pkl"  # <--- Nom du fichier de base de données locale

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 20
DEFAULT_MIN_SCORE = 0.2
logging.basicConfig(level=logging.INFO)
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
    "Tu es EcoBot 🌿, un assistant expert en organisation d'événements éco-responsables. "
    "Tu es rigoureux, structuré et obsédé par la traçabilité de l'information."
    "\n\n"
    "TES OBJECTIFS :"
    "\n1. UTILISATION DU CONTEXTE : Réponds EXCLUSIVEMENT à partir du 'CONTEXTE DOCUMENTAIRE' fourni. "
    "Les extraits sont numérotés (ex: [1], [2])."
    "\n2. CITATIONS OBLIGATOIRES : Pour chaque affirmation importante, tu DOIS indiquer le numéro de la source entre crochets."
    "\n   - Exemple : 'Les gobelets jetables sont interdits [1], privilégiez les Ecocups consignés [2].'"
    "\n3. FORMATAGE : Utilise des Titres (##), des listes à puces (-) et du **gras** pour la lisibilité."
    "\n4. DÉTAILS : Sois complet et précis. Si l'utilisateur demande un plan, fournis un plan détaillé."
    "\n5. 🔍 CITATIONS OBLIGATOIRES : Chaque affirmation doit être sourcée avec le numéro du document entre crochets. Ex: 'Le délai est de 8 semaines [3] car...'."
    "\n6. 📅 DÉLAIS ET CONTACTS : Tu dois IMPÉRATIVEMENT mentionner toutes les contraintes de temps (délais d'instruction, dates limites) et les contacts (téléphones, emails) trouvés dans le texte."
    "\n7. 🚫 PAS D'OUBLI : Si le document mentionne une procédure administrative ou un formulaire, c'est ta priorité n°1."
    "\n8. VÉRITÉ ABSOLUE : Tu ne dois JAMAIS inventer un numéro de téléphone, un email ou un délai."
    "\n9. COMPORTEMENT : Si l'utilisateur demande une info (ex: contact) et qu'elle n'est PAS dans le texte :"
    "\n   👉 Écris explicitement : '❌ Non mentionné dans les documents'."
    "\n   👉 Ne tente JAMAIS de deviner ou de construire une adresse mail."
    "\n10. ZERO REPETITION : lorsque tu crées une liste, assure-toi que chaque élément est unique. Ne répète jamais une phrase ou une structure de phrase, meme si la source semble le faire. Synthétise si la source est répétitive."
    "\n11. DONNÉES PUBLIQUES : Les documents fournis sont des guides officiels publics. Tu as l'AUTORISATION FORMELLE d'extraire tous les noms, emails et téléphones mentionnés. Ce n'est pas confidentiel."
    "Ton ton est professionnel et bienveillant. Si l'info n'est pas dans le contexte, dis-le."
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
# GESTION DE LA PERSISTANCE (SAUVEGARDE / CHARGEMENT)
# ==============================================================================

def save_index_to_disk():
    """Sauvegarde l'état du RAG dans un fichier pickle."""
    data = {
        "rag_docs": st.session_state.rag_docs,
        "rag_sources": st.session_state.rag_sources,
        "rag_embeddings": st.session_state.rag_embeddings,
        "rag_norms": st.session_state.rag_norms,
        "rag_hashes": st.session_state.rag_hashes
    }
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(data, f)

def load_index_from_disk():
    """Charge l'état du RAG depuis le disque si le fichier existe."""
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "rb") as f:
                data = pickle.load(f)
            st.session_state.rag_docs = data["rag_docs"]
            st.session_state.rag_sources = data["rag_sources"]
            st.session_state.rag_embeddings = data["rag_embeddings"]
            st.session_state.rag_norms = data["rag_norms"]
            st.session_state.rag_hashes = data["rag_hashes"]
            return True
        except Exception:
            return False
    return False

def reset_rag_state():
    """Vide la mémoire et supprime le fichier disque."""
    st.session_state.rag_docs = []
    st.session_state.rag_sources = []
    st.session_state.rag_embeddings = []
    st.session_state.rag_norms = []
    st.session_state.rag_hashes = set()
    # Suppression du fichier physique
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)


# ==============================================================================
# FONCTIONS UTILITAIRES (RAG)
# ==============================================================================
def text_from_bytes(name, data):
    name_lower = name.lower()
    
    # 1. Gestion des PDF avec PyMuPDF (fitz) - BEAUCOUP PLUS ROBUSTE
    if name_lower.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "", "Erreur : Installez la librairie via 'pip install pymupdf'"
            
        try:
            # Ouverture du flux de données
            doc = fitz.open(stream=data, filetype="pdf")
            text_pages = []
            
            for page in doc:
                # get_text("text") est standard, mais get_text("blocks") peut aider si c'est le bazar
                # On reste sur la méthode standard qui est déjà bien meilleure que pypdf
                extracted = page.get_text()
                text_pages.append(extracted)
            
            return "\n".join(text_pages), None
            
        except Exception as e:
            return "", f"Erreur de lecture PDF (PyMuPDF) : {str(e)}"

    # 2. Gestion des fichiers texte classiques (txt, md)
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
    # 1. Calcul du vecteur de la question
    query_embedding = embed_texts([query], client)[0]
    query_norm = vector_norm(query_embedding)
    
    if query_norm == 0:
        return [], []

    # 2. Préparation des mots-clés (Hybrid Search Frugale)
    # On découpe la question en mots, on met en minuscule, et on garde ceux > 3 lettres
    # Ex: "téléphone service gestion" -> ['téléphone', 'service', 'gestion']
    query_keywords = [w.lower() for w in query.split() if len(w) > 3]

    scored = []
    for idx, embedding in enumerate(st.session_state.rag_embeddings):
        # A. Score Vectoriel (Le Sens)
        denom = query_norm * st.session_state.rag_norms[idx]
        cosine_score = dot_product(query_embedding, embedding) / denom if denom else 0.0
        
        # B. Score Mots-Clés (Le Bonus "Exactitude")
        # On regarde le texte brut du chunk
        chunk_text_lower = st.session_state.rag_docs[idx].lower()
        keyword_bonus = 0.0
        
        for kw in query_keywords:
            if kw in chunk_text_lower:
                # CHAQUE mot clé trouvé ajoute +5% de pertinence
                keyword_bonus += 0.05
        
        # On limite le bonus à +30% max pour ne pas casser complètement la logique sémantique
        keyword_bonus = min(keyword_bonus, 0.30)
        
        # C. Score Final Combiné
        final_score = cosine_score + keyword_bonus
        
        scored.append((final_score, idx))

    # 3. Tri des résultats
    scored.sort(key=lambda item: item[0], reverse=True)
    
    chunks = []
    sources = set()
    
    # On récupère les meilleurs
    for score, idx in scored[:top_k]:
        # On filtre si le score est vraiment trop bas (même avec bonus)
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
    # On initialise vide
    st.session_state.rag_docs = []
    st.session_state.rag_sources = []
    st.session_state.rag_embeddings = []
    st.session_state.rag_norms = []
    st.session_state.rag_hashes = set()
    
    # On essaie de charger le disque
    if load_index_from_disk():
        st.toast(f"♻️ Index chargé : {len(st.session_state.rag_docs)} chunks restaurés !", icon="💾")

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

# Note sur la persistance
if os.path.exists(INDEX_FILE):
    st.sidebar.success("💾 Une base de données existe.")
else:
    st.sidebar.warning("⚪ Aucune base sauvegardée.")

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
        max_value=50,
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

    if st.button("🗑️ Tout effacer (RAM + Disque)"):
        reset_rag_state()
        st.rerun()


# TRAITEMENT DES FICHIERS UPLOADÉS
if uploaded_files:
    new_data_added = False
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
        # if use_gdpr_shield:
        #     with st.spinner(f"🛡️ Anonymisation IA en cours : {uploaded_file.name}"):
        #         text, count_pii = gdpr_shield.anonymize(text)
        #     if count_pii > 0:
        #         st.sidebar.warning(f"🛡️ {count_pii} éléments masqués dans {uploaded_file.name}")
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
        new_data_added = True
        st.sidebar.success(
            f"Indexe: {uploaded_file.name} ({len(chunks)} chunks)"
        )

     # SAUVEGARDE AUTOMATIQUE APRÈS L'AJOUT
    if new_data_added:
        save_index_to_disk()
        st.toast("Index mis à jour et sauvegardé sur disque !", icon="✅")


# Affichage des sources
if st.session_state.rag_sources:
    st.sidebar.markdown(f"**{len(set(st.session_state.rag_sources))} documents** en mémoire.")
    with st.sidebar.expander("Voir la liste"):
        for name in sorted(set(st.session_state.rag_sources)):
            st.write(f"- {name}")
else:
    st.sidebar.info("Aucun document.")

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
        logging.info("Anonymisation du prompt utilisateur...")
        logging.info(f"Prompt avant: {safe_prompt}")
        with st.spinner("Anonymisation de votre question..."):
            safe_prompt, count_prompt_pii = gdpr_shield.anonymize(prompt)
        logging.info(f"Prompt apres: {safe_prompt}")
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
                    safe_prompt, top_k, min_score, client
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

    # --- AJOUT DEBUG : VOIR CE QUE L'IA LIT ---
    with st.expander("🔍 DEBUG : Voir le Contexte envoyé au LLM", expanded=False):
        st.markdown(f"**Nombre de documents lus :** {len(retrieved_chunks)}")
        st.text_area("Contenu brut envoyé", final_content, height=300)
    # ------------------------------------------

    # 7. APPEL LLM EN STREAMING
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_for_llm,
            stream=True,
            # --- AJOUTS ANTI-BUG ---
            temperature=0.3,       # On baisse la créativité (évite qu'il parte en vrille)
            top_p=0.9,             # Noyau de probabilité
            presence_penalty=0.6,  # Punit si le mot est déjà là (évite la redite)
            frequency_penalty=1.5, # Punit si le mot apparaît trop souvent
            max_tokens=2048
        )
        response = st.write_stream(stream)
        if used_sources:
            st.caption("Sources: " + ", ".join(used_sources))

    st.session_state.messages.append({"role": "assistant", "content": response})

# =========================================================
# OUTIL DE DIAGNOSTIC (A AJOUTER A LA FIN DU FICHIER)
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("🛠️ DIAGNOSTIC PDF")
debug_query = st.sidebar.text_input("Chercher un mot EXACT dans la mémoire :", "")

if debug_query:
    found_count = 0
    st.sidebar.write(f"Recherche de : '{debug_query}'")
    
    # On parcourt TOUS les morceaux en mémoire
    for i, doc in enumerate(st.session_state.rag_docs):
        # On cherche le mot (insensible à la casse)
        if debug_query.lower() in doc.lower():
            found_count += 1
            st.sidebar.success(f"TROUVÉ dans le Chunk #{i}")
            with st.sidebar.expander(f"Voir le contenu du Chunk #{i}"):
                # On surligne le mot trouvé
                highlighted = doc.replace(debug_query, f"**{debug_query}**")
                st.markdown(highlighted)
    
    if found_count == 0:
        st.sidebar.error("❌ CE MOT N'EXISTE PAS DANS LA MÉMOIRE.")
        st.sidebar.warning("Conclusion : pypdf a mal lu le fichier ou le mot est mal écrit/coupé.")