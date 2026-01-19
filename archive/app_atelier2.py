import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import time
import av
import cv2
import re

import os
import sys
from pathlib import Path

from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, pipelines

# --- FIX LOGISTIQUE : CHARGEMENT DES DLL LIBVIPS (MODE PORTABLE) ---
# On utilise __file__ pour être sûr de partir de l'emplacement du script
BASE_DIR = Path(__file__).parent.absolute()
VIPS_BIN_PATH = os.path.join(BASE_DIR, "vips-8.16", "bin")

if os.path.exists(VIPS_BIN_PATH):
    # 1. On l'ajoute au PATH système pour les sous-dépendances
    os.environ['PATH'] = VIPS_BIN_PATH + os.pathsep + os.environ['PATH']
    # 2. On force Python 3.12 à accepter ce dossier pour les DLL
    os.add_dll_directory(VIPS_BIN_PATH)
    print(f"✅ Libvips chargé depuis : {VIPS_BIN_PATH}")
else:
    # Ce message apparaîtra dans la console noire si le dossier n'est pas trouvé
    print(f"❌ ERREUR : Dossier libvips introuvable à {VIPS_BIN_PATH}")


# --- CONFIGURATION ---
st.set_page_config(page_title="IA Frugale", page_icon="✂️", layout="wide")

# --- INITIALISATION ---
if 'water' not in st.session_state: st.session_state.water = 0.0
if 'co2' not in st.session_state: st.session_state.co2 = 0.0
if 'last_res' not in st.session_state: st.session_state.last_res = None

# --- LA FONCTION MAGIQUE (Maintenant elle existe vraiment !) ---
def update_impact(ml, g):
    """Met à jour les économies d'eau et de CO2 dans la session"""
    st.session_state.water += ml
    st.session_state.co2 += g

# --- CSS HAUTE VISIBILITÉ (Texte blanc pur sur fond noir) ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .impact-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #00CCFF;
        text-align: center;
        margin-bottom: 20px;
    }
    .impact-label { color: #FFFFFF !important; font-weight: bold; font-size: 16px; margin-bottom: 0px; }
    .big-num { font-size: 32px; font-weight: bold; color: #00CCFF !important; margin-top: 0px; }
    
    .ds-container { display: flex; gap: 8px; margin: 20px 0; align-items: center; }
    .ds-box { padding: 10px 18px; border-radius: 6px; font-weight: bold; color: #333; background: #ddd; }
    .ds-active { color: white !important; box-shadow: 0 0 15px rgba(255,255,255,0.2); }
    .grade-a { background-color: #008000 !important; }
    .grade-b { background-color: #80FF00 !important; }
    </style>
    """, unsafe_allow_html=True)

def render_digiscore(grade):
    html = '<div class="ds-container"><span style="font-weight:bold;">Indice Frugalité :</span>'
    for g in ['A', 'B', 'C', 'D', 'E']:
        active = f"ds-active grade-{g.lower()}" if g == grade else ""
        html += f'<div class="ds-box {active}">{g}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_models():
    # On fixe une révision stable pour éviter les mauvaises surprises
    MD_REVISION = "2025-01-09" 

    return {
        "gliner": GLiNER.from_pretrained("urchade/gliner_small-v2.1"),
        "embedder": SentenceTransformer('all-MiniLM-L6-v2'),
        "squeezer": pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct"),
        "vision": AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", 
            trust_remote_code=True, 
            revision=MD_REVISION
        )
    }

models = load_models()
# 1. Définition de la fonction de chargement (avec cache pour la performance)
@st.cache_resource
def load_llm():
    """
    Charge un modèle très léger (SmolLM2-135M) pour une exécution locale rapide.
    Le cache permet de ne le charger qu'une seule fois en mémoire.
    """
    # On utilise SmolLM2-135M-Instruct, parfait pour des démos CPU rapides
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    return pipe

# --- MOTEUR D'ANONYMISATION ROBUSTE ---
def anonymize_text(text, model):
    # 1. On définit TOUT ce qu'on veut que l'IA trouve
    labels = ["person", "location", "organization", "date", "job title", "amount", "address", "phone number"]
    entities = model.predict_entities(text, labels, threshold=0.3)
    
    # 2. On ajoute des Regex pour les structures fixes (Sécu, IBAN, Mails)
    patterns = {
        "SOCIAL_SECURITY": r'[12][ ]?[0-9]{2}[ ]?[0-1][0-9][ ]?[0-9]{2,3}[ ]?[0-9]{3}[ ]?[0-9]{3}[ ]?[0-9]{2}',
        "IBAN": r'FR[0-9]{2}[ ]?([0-9]{4}[ ]?){5}[0-9]{3}',
        "EMAIL": r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
        "PHONE": r'(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}'
    }
    
    # On compile toutes les zones à cacher (NER + Regex)
    spans = []
    for ent in entities:
        spans.append((ent['start'], ent['end'], ent['label'].upper()))
    
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), label))

    # 3. Tri et fusion des zones pour éviter les chevauchements (le bug du I[EMAIL]ement)
    spans.sort(key=lambda x: x[0])
    merged_spans = []
    if spans:
        curr_start, curr_end, curr_label = spans[0]
        for next_start, next_end, next_label in spans[1:]:
            if next_start < curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged_spans.append((curr_start, curr_end, curr_label))
                curr_start, curr_end, curr_label = next_start, next_end, next_label
        merged_spans.append((curr_start, curr_end, curr_label))

    # 4. Remplacement de la fin vers le début pour garder les index valides
    result = text
    for start, end, label in reversed(merged_spans):
        result = result[:start] + f"[{label}]" + result[end:]
    return result


# --- SIDEBAR IMPACT ---
st.sidebar.title("🌍 Impact Planétaire")
st.sidebar.markdown(f"""
<div class="impact-card">
    <p class="impact-label">💧 Eau économisée</p>
    <p class="big-num">{round(st.session_state.water, 1)} ml</p>
    <p class="impact-label">☁️ CO2 évité</p>
    <p class="big-num">{round(st.session_state.co2, 2)} g</p>
</div>
""", unsafe_allow_html=True)

tool = st.sidebar.radio("Modules :", ["🛡️ GDPR Shield", "🧠 Brain Map", "🎨 Sketch2Code", "📉 Token Squeezer", "🖐️ Hand Control"])

# Reset si changement d'onglet
if 'current_tool' not in st.session_state or st.session_state.current_tool != tool:
    st.session_state.current_tool = tool
    st.session_state.last_res = None

# =========================================================
# MODULE 1 : GDPR SHIELD
# =========================================================
if tool == "🛡️ GDPR Shield":
    st.header("🛡️ GDPR Shield : Anonymisation Forteresse")
    render_digiscore("A")
    
    sample_text = """M. Jean Martin, né le 3 septembre 1979 à Bordeaux, réside au 42 rue de la Paix, 33100 Bordeaux. 
Il est marié à Claire Lefèvre. Son IBAN est FR76 3000 4000 1200 5678 9012 345.
Responsable logistique chez LogiTrans SA, son salaire est de 48 000 euros.
Numéro de sécu : 1 79 09 33 456 789 01. Contact : 06 88 21 45 09."""

    text_in = st.text_area("Texte sensible :", sample_text, height=250)
    
    if st.button("Lancer l'extraction locale"):
        with st.spinner("Analyse sémantique et filtrage local..."):
            res = anonymize_text(text_in, models['gliner'])
            st.session_state.last_res = res
            st.session_state.water += 500
            st.session_state.co2 += 2.5
        st.rerun()
    
    if st.session_state.last_res:
        st.markdown(f'<div class="result-area">{st.session_state.last_res}</div>', unsafe_allow_html=True)

# =========================================================
# MODULE 2 : Brain Map
# =========================================================
elif tool == "🧠 Brain Map":
    st.header("🧠 Brain Map : Intelligence Structurelle")
    render_digiscore("B")
    # L'INFO-CHOC (À afficher tout de suite)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Modèle Local (Le vôtre)", "80 Mo", delta="Léger", delta_color="normal")
    with col_b:
        st.metric("Modèle Cloud (Standard)", "350 000 Mo", delta="4300x plus gros", delta_color="inverse")

    st.warning("⚠️ Pour trier vos 4 lignes, le Cloud mobilise un cerveau 4000 fois plus gros que nécessaire.")
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    # --- LE CURSEUR DE RÉGLAGE (La finesse) ---
    st.markdown("### 🛠️ Réglage de la finesse")
    threshold = st.slider(
        "Sensibilité du regroupement (Threshold) :", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Plus le seuil est BAS, plus l'IA est sévère et crée de nombreux petits groupes précis. Plus il est HAUT, plus elle mélange les documents."
    )
    
    # Petit indicateur visuel pour aider l'utilisateur
    if threshold < 0.4:
        st.caption("🔍 **Mode Chirurgical** : Idéal pour séparer des documents très proches (ex: deux types de contrats).")
    elif threshold > 0.7:
        st.caption("📦 **Mode Global** : Idéal pour voir les grandes masses (ex: Pro vs Perso).")
    else:
        st.caption("⚖️ **Mode Équilibré** : Le réglage standard.")


    default_docs = (
        "Facture électricité Janvier\n"
        "Facture gaz Février\n"
        "Régularisation eau 2023\n"
        "Recette de cuisine : Tarte aux pommes\n"
        "Préparation culinaire : Crêpes bretonnes\n"
        "Ingrédients gâteau au chocolat\n"
        "Devis rénovation toiture\n"
        "Estimation isolation murs\n"
        "Contrat de travail CDD\n"
        "Avenant mutuelle entreprise\n"
        "Demande de congés payés"
    )

    raw_data = st.text_area("📦 Documents à classer :", default_docs, height=200, key="t2")
    
    if st.button("Organiser intelligemment"):
        # On remplace les mots ambigus pour aider le petit modèle frugal
        docs = [d.strip() for d in raw_data.split('\n') if d.strip()]
        # Petit hack frugal pour lever l'ambiguïté sur "Recette"
        docs_for_ai = [d.replace("Recette", "Cuisine recette").replace("Facture", "Document comptable facture") for d in docs]
        
        if len(docs) > 2:
            with st.spinner("Analyse des vecteurs sémantiques..."):
                # 1. Vectorisation
                embeddings = models['embedder'].encode(docs_for_ai)
                # 2. Normalisation (Crucial pour la précision sémantique)
                embeddings = normalize(embeddings)
                
                # 3. Clustering Hiérarchique (plus précis que KMeans pour le texte)
                # On utilise une distance cosinus pour ignorer la taille des phrases
                cluster_model = AgglomerativeClustering(
                    n_clusters=None, 
                    distance_threshold=threshold, # Plus c'est bas, plus il crée de petits groupes précis
                    metric='cosine', 
                    linkage='complete'
                )
                cluster_labels = cluster_model.fit_predict(embeddings)
                num_groups = len(set(cluster_labels))
                
                # 4. Projection 2D
                pca = PCA(n_components=2).fit_transform(embeddings)
                
                update_impact(300, 1.5)
                
                df = {
                    'x': pca[:, 0], 'y': pca[:, 1],
                    'Document': docs,
                    'Thématique': [f"Thème {l+1}" for l in cluster_labels]
                }
                
                fig = px.scatter(
                    df, x='x', y='y', text='Document', color='Thématique',
                    title=f"Organisation en {num_groups} thématiques distinctes",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_traces(textposition='top center', marker=dict(size=14, line=dict(width=1, color='white')))
                fig.update_layout(dragmode='pan')
                fig.update_xaxes(showticklabels=False, title_text="Similitude thématique →")
                fig.update_yaxes(showticklabels=False, title_text="Différence de contexte ↑")
                
                st.session_state.last_res = {"fig": fig, "df": df}
            st.rerun()

    if st.session_state.last_res:
        st.plotly_chart(st.session_state.last_res["fig"], use_container_width=True)
        
        # Affichage des colonnes dynamique selon le nombre de thèmes trouvés
        res_df = st.session_state.last_res["df"]
        themes = sorted(set(res_df['Thématique']))
        
        st.markdown("### 📋 Classement automatique")
        # On crée des colonnes (max 4 par ligne)
        for i in range(0, len(themes), 4):
            cols = st.columns(min(4, len(themes)-i))
            for j, theme in enumerate(themes[i:i+4]):
                with cols[j]:
                    st.info(f"**{theme}**")
                    items = [res_df['Document'][k] for k, t in enumerate(res_df['Thématique']) if t == theme]
                    for item in items:
                        st.write(f"• {item}")

# =========================================================
# MODULE 3 : Sketch2Code
# =========================================================
elif tool == "🎨 Sketch2Code":
    st.header("🎨 Vision Frugale : Le dessin analysé")
    render_digiscore("B")
    
    st.markdown("""
    **Leçon :** Un petit modèle local est parfait pour décrire une scène ou une personne. 
    Mais attention : sur des schémas très complexes, le niveau de précision atteint toutefois ses limites.
    """)

    file = st.file_uploader("Image / Croquis", type=['png', 'jpg'], key="u3")
    if file:
        img = Image.open(file)
        st.image(img, width=400)
        
        # On suggère des questions plus précises
        q_default = "Décris cette image précisément en français."
        prompt = st.text_input("Votre question :", q_default, key="p3")
        
        if st.button("Analyser l'image"):
            with st.spinner("L'IA locale analyse les pixels..."):
                update_impact(800, 5.0)
                
                 # 1. On prépare le tokenizer (nécessaire pour le contrôle fin)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2025-01-09")

                full_prompt = f"Identify and list all UI elements, colors, and icons in this image. Answer in French."

                # 3. On décompose l'appel pour ajouter les paramètres de puissance
                enc_image = models['vision'].encode_image(img)
                
                # C'EST ICI QUE ÇA CHANGE :
                answer = models['vision'].answer_question(
                    enc_image, 
                    full_prompt, 
                    tokenizer,
                    max_new_tokens=300,  # On lui donne le droit de parler longtemps (300 mots max)
                    iteration_count=3,   # On le force à regarder l'image plusieurs fois
                )
                
                # Appel sécurisé
                try:
                    # On demande une réponse un peu plus longue
                    result = models['vision'].query(img, full_prompt)
                    answer = result["answer"]
                    
                    # Si l'IA est têtue et répond en anglais, on peut ajouter une note
                    st.session_state.last_res = answer
                except Exception as e:
                    st.session_state.last_res = f"Erreur d'analyse : {str(e)}"
            st.rerun()

    if st.session_state.last_res:
        st.subheader("Analyse de l'IA :")
        # On force l'affichage en français si possible
        st.info(st.session_state.last_res)

# =========================================================
# MODULE 4 : TOKEN SQUEEZER (TEXT GEN)
# =========================================================
elif tool == "📉 Token Squeezer":
    st.header("📉 Token Squeezer (V3 - Mode Few-Shot)")
    st.caption("Stratégie : Donner des exemples au modèle pour forcer la brièveté.")

    user_prompt = st.text_area("Votre prompt verbeux :", 
                            "Je voudrais que tu agisses comme un expert en marketing et que tu m'écrives un post pour LinkedIn qui parle de l'IA frugale, il faut que ce soit court, percutant, avec des emojis, et que ça explique pourquoi c'est écolo.")

    if st.button("Compresser le Prompt"):
        with st.spinner("Chargement..."):
            generator = load_llm()
        
    # --- LA MAGIE EST ICI : FEW-SHOT PROMPTING ---
    # On donne des exemples "Avant -> Après" pour forcer le modèle à imiter le style
    system_instruction = """Tu es un expert en compression de texte. 
    Ta tâche : Transformer des demandes longues en commandes impératives courtes.
    Règles : Supprime la politesse. Supprime 'Je veux que'. Utilise l'impératif. Pas de listes.

    Exemple 1 :
    Entrée : "Je voudrais que tu agisses comme un coach sportif et que tu me donnes un plan pour perdre du poids."
    Sortie : "Agis comme un coach sportif. Crée un plan de perte de poids."

    Exemple 2 :
    Entrée : "Peux-tu écrire un poème sur la pluie en style victorien s'il te plait ?"
    Sortie : "Écris un poème victorien sur la pluie."

    Exemple 3 :
    Entrée : "Je veux une explication simple de la quantique pour un enfant de 5 ans."
    Sortie : "Explique la physique quantique à un enfant de 5 ans."
    """
    
    # On construit le message final
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Entrée : \"{user_prompt}\"\nSortie :"}
    ]
    
    with st.spinner("Compression drastique..."):
        # Max token réduit pour couper la parole s'il devient bavard
        result = generator(messages, max_new_tokens=60, temperature=0.1) 
        
        output = result[0]['generated_text'][-1]['content']
        
        # Nettoyage final (parfois il laisse des guillemets)
        output = output.replace('"', '').strip()

        # Calculs
        len_original = len(user_prompt.split())
        len_optimized = len(output.split())
        reduction = ((len_original - len_optimized) / len_original) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.warning(f"Original ({len_original} mots)")
            st.write(user_prompt)
        with col2:
            st.success(f"Optimisé ({len_optimized} mots, -{int(reduction)}%)")
            st.code(output, language="text")

# =========================================================
# MODULE 5 : HAND CONTROL
# =========================================================
elif tool == "🖐️ Hand Control":
    st.header("🖐️ L'IA sans serveur")
    render_digiscore("A")
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    
    class HandProcessor(VideoProcessorBase):
        def __init__(self): self.h = mp.solutions.hands.Hands(model_complexity=0)
        def recv(self, frame):
            img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
            results = self.h.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(img, lm, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.putText(img, "LOCAL - 0 WATER", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="hands", video_processor_factory=HandProcessor)
