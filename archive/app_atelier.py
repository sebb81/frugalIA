import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)
import av

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="La Boîte à Outils Frugale", page_icon="🧰", layout="wide")

st.title("🧰 La Boîte à Outils Frugale")
st.markdown("""
Cette application démontre qu'on peut résoudre des problèmes complexes avec des modèles **locaux, petits et spécialisés**.
Choisissez un outil dans le menu de gauche.
""")

# --- SIDEBAR ---
tool = st.sidebar.radio(
    "Choisir un module :",
    ["🛡️ GDPR Shield (Anonymisation)", 
     "🧠 Brain Map (Classement Sémantique)", 
     "🎨 Sketch2Code (Vision)", 
     "📉 Token Squeezer (Optimisation)",
     "🖐️ Hand Control (Gestes)"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"🖥️ Exécution sur : {'GPU' if torch.cuda.is_available() else 'CPU'}")

# --- FONCTIONS DE CHARGEMENT (LAZY LOADING) ---
# On utilise @st.cache_resource pour ne charger les modèles qu'une seule fois

@st.cache_resource
def load_gliner():
    from gliner import GLiNER
    # Modèle spécialisé NER, ultra léger (~180MB)
    return GLiNER.from_pretrained("urchade/gliner_small-v2.1")

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    # Modèle de vectorisation rapide (~80MB)
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_vision_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from transformers.generation.utils import GenerationMixin
    # Moondream2 : Le roi de la vision sur CPU (~2GB)
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    if hasattr(model, "text_model") and not isinstance(model.text_model, GenerationMixin):
        model.text_model.__class__ = type(
            "PhiForCausalLMWithGen",
            (GenerationMixin, model.text_model.__class__),
            {},
        )
    if hasattr(model, "text_model") and not hasattr(model.text_model, "generation_config"):
        model.text_model.generation_config = GenerationConfig.from_model_config(
            model.text_model.config
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

@st.cache_resource
def load_llm():
    from transformers import pipeline
    # Un tout petit modèle de texte pour l'exemple (ex: GPT-2 ou un TinyLlama)
    # Pour un vrai atelier, on préférera un modèle GGUF via llama-cpp, 
    # mais ici on utilise transformers pour la simplicité du code.
    return pipeline("text-generation", model="openai/gpt-oss-20b", device_map="auto")


# =========================================================
# MODULE 1 : GDPR SHIELD (ANONYMISATION)
# =========================================================
if tool == "🛡️ GDPR Shield (Anonymisation)":
    st.header("🛡️ GDPR Shield : Anonymisation Intelligente")
    st.caption("Leçon : Un petit modèle spécialisé (GLiNER) bat un gros LLM sur l'extraction d'entités.")
    
    text_input = st.text_area("Collez un texte sensible ici (Email, Rapport médical...)", 
                              "M. Dupont habite au 15 rue de la Paix à Paris et son numéro est le 06 12 34 56 78. Il souffre de diabète.")
    
    labels = st.multiselect("Quoi cacher ?", ["person", "location", "phone number", "disease", "iban"], default=["person", "location", "phone number"])

    if st.button("Anonymiser"):
        with st.spinner("Chargement de GLiNER (180 Mo)..."):
            model = load_gliner()
            
        with st.spinner("Analyse en cours..."):
            entities = model.predict_entities(text_input, labels)
            
            anonymized_text = text_input
            # On trie par ordre décroissant pour ne pas casser les index lors du remplacement
            for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
                replacement = f"[{entity['label'].upper()}]"
                start, end = entity['start'], entity['end']
                anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.text(text_input)
            with col2:
                st.subheader("Anonymisé")
                st.success(anonymized_text)
            
            st.write("🔍 **Entités détectées :**", entities)

# =========================================================
# MODULE 2 : BRAIN MAP (CLUSTERING SÉMANTIQUE)
# =========================================================
elif tool == "🧠 Brain Map (Classement Sémantique)":
    st.header("🧠 Brain Map : Organisation de documents")
    st.caption("Leçon : Les vecteurs (Embeddings) coûtent 100x moins cher que la génération de texte.")

    st.info("Entrez plusieurs phrases (une par ligne) pour voir comment l'IA les regroupe par sens.")
    
    default_corpus = """Facture électricité Janvier
Recette de la tarte aux pommes
Devis rénovation toiture
Ingrédients pour crêpes
Facture gaz Février
Contrat assurance auto
Comment cuire un steak
Bon de commande fournisseur"""
    
    corpus = st.text_area("Vos documents (1 par ligne)", default_corpus, height=200).split('\n')
    corpus = [doc for doc in corpus if doc.strip() != ""] # Nettoyage

    if st.button("Générer la Carte"):
        with st.spinner("Vectorisation avec all-MiniLM..."):
            embedder = load_embedder()
            embeddings = embedder.encode(corpus)
            
            # Réduction de dimension (384 dim -> 2 dim) pour l'affichage
            pca = PCA(n_components=2)
            vis_dims = pca.fit_transform(embeddings)
            
            # Clustering simple (K-Means)
            n_clusters = min(3, len(corpus))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            # Création du graph
            fig = px.scatter(
                x=vis_dims[:, 0], y=vis_dims[:, 1],
                text=corpus, color=clusters.astype(str),
                title="Carte Sémantique des Documents",
                labels={'color': 'Groupe'}
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODULE 3 : SKETCH2CODE (VISION)
# =========================================================
elif tool == "🎨 Sketch2Code (Vision)":
    st.header("🎨 Sketch2Code : Du croquis au Code")
    st.caption("Leçon : La multimodalité sur CPU est possible avec des modèles comme Moondream.")

    uploaded_file = st.file_uploader("Uploadez un croquis d'interface ou une photo", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image source", width=300)
        
        prompt = st.text_input("Question pour l'IA", "Décris cette image en détail pour un développeur web.")
        
        if st.button("Analyser"):
            with st.spinner("Chargement de Moondream (peut être lent la 1ère fois)..."):
                model, tokenizer = load_vision_model()
            
            with st.spinner("L'IA regarde l'image..."):
                enc_image = model.encode_image(image)
                answer = model.answer_question(enc_image, prompt, tokenizer)
                st.markdown("### Réponse de l'IA :")
                st.code(answer, language="markdown")

# =========================================================
# MODULE 4 : TOKEN SQUEEZER (TEXT GEN)
# =========================================================
elif tool == "📉 Token Squeezer (Optimisation)":
    st.header("📉 Token Squeezer")
    st.caption("Leçon : Utiliser une petite IA gratuite pour optimiser les prompts d'une grosse IA payante.")

    user_prompt = st.text_area("Votre prompt verbeux :", 
                               "Je voudrais que tu agisses comme un expert en marketing et que tu m'écrives un post pour LinkedIn qui parle de l'IA frugale, il faut que ce soit court, percutant, avec des emojis, et que ça explique pourquoi c'est écolo.")

    if st.button("Compresser le Prompt"):
        with st.spinner("Chargement du petit LLM (SmolLM)..."):
            generator = load_llm()
        
        system_instruction = "Tu es un expert en Prompt Engineering. Réécris la demande suivante pour qu'elle soit concise, directe et efficace pour un LLM : "
        full_prompt = f"{system_instruction}\n\nDemande originale: {user_prompt}\n\nPrompt optimisé:"
        
        with st.spinner("Réécriture en cours..."):
            # On utilise un petit modèle (SmolLM) pour la démo rapide sans download de 5Go
            messages = [
                {"role": "user", "content": full_prompt},
            ]
            result = generator(messages, max_new_tokens=150, temperature=0.7)
            
            output = result[0]['generated_text'][-1]['content'] # Récupérer juste la réponse si format chat
            
            col1, col2 = st.columns(2)
            with col1:
                st.warning(f"Prompt original ({len(user_prompt.split())} mots)")
            with col2:
                st.success(f"Prompt optimisé")
                st.write(output)

# =========================================================
# MODULE 5 : HAND CONTROL (TEMPS RÉEL + GESTES SIMPLES)
# =========================================================
elif tool == "🖐️ Hand Control (Gestes)":
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

    st.header("🖐️ Hand Control : Perception Temps Réel")
    st.caption("Leçon : MediaPipe traite ~30 FPS sur CPU, sans gros modèle.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.info("Activez la caméra, faites un signe (✌️, ✋, ☝️, 👍) et observez la détection en temps réel.")
        run = st.checkbox('🔴 Activer la Caméra', value=False)
        kpi_placeholder = st.empty()
        gesture_placeholder = st.empty()

    with col2:
        st.write("Aperçu en direct :")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    class HandTracker(VideoProcessorBase):
        def __init__(self) -> None:
            self.hands = mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,  # plus léger pour le temps réel
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.status_text = "En attente de main..."
            self.gesture_text = "Aucun geste détecté"

        def _count_fingers(self, hand_landmarks, handedness_label):
            count = 0

            # Index, majeur, annulaire, auriculaire
            tips = [8, 12, 16, 20]
            for tip in tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    count += 1

            # Pouce
            thumb_tip  = hand_landmarks.landmark[4]
            thumb_ip   = hand_landmarks.landmark[3]
            thumb_mcp  = hand_landmarks.landmark[2]

            if handedness_label == "Right":
                # main droite : pouce vers la gauche
                if thumb_tip.x < thumb_ip.x < thumb_mcp.x:
                    count += 1
            else:  # "Left"
                # main gauche : pouce vers la droite
                if thumb_tip.x > thumb_ip.x > thumb_mcp.x:
                    count += 1

            return count

        def _gesture_from_fingers(self, total_fingers):
            """
            Mapping simple nombre de doigts -> geste symbolique.
            On peut raffiner plus tard (pattern par doigt).
            """
            if total_fingers == 5:
                return "Geste detecte : STOP / Bonjour"
            if total_fingers == 2:
                return "Geste detecte : Paix"
            if total_fingers == 1:
                return "Geste detecte : 1"
            if total_fingers == 0:
                return "Geste detecte : Poing ferme"
            if total_fingers == 3:
                return "Geste detecte : 3 (ou signe rock)"
            # fallback
            return f"Aucun geste clair (doigts : {total_fingers})"

        def recv(self, frame):
            # Frame -> ndarray OpenCV
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)
            image.flags.writeable = False  # hint perf pour cv2/mediapipe

            # MediaPipe veut du RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            image.flags.writeable = True

            status_text = "En attente de main..."
            gesture_text = "Aucun geste detecte"

            if results.multi_hand_landmarks:
                total_fingers = 0

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, 
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" ou "Right"

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )

                    total_fingers += self._count_fingers(hand_landmarks, label)

                status_text = f"Doigts leves (total) : {total_fingers}"
                gesture_text = self._gesture_from_fingers(total_fingers)

            self.status_text = status_text
            self.gesture_text = gesture_text

             # 🔹 Overlay texte directement sur la frame
            cv2.putText(
                image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                gesture_text,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    if run:
        ctx = webrtc_streamer(
            key="hand-control",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HandTracker,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": {"ideal": 15, "max": 20}},
                "audio": False,
            },
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            async_processing=True,
        )

        if ctx.video_processor:
            kpi_placeholder.markdown(f"### 📊 {ctx.video_processor.status_text}")
            gesture_placeholder.markdown(f"### 🤟 {ctx.video_processor.gesture_text}")
        elif ctx.state.playing is False:
            st.warning("Cliquez sur 'Start' dans le composant vidéo pour autoriser la caméra.")
    else:
        st.warning("Activez la caméra pour démarrer la détection des gestes.")
