import streamlit as st
import ollama
from PIL import Image
import io

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Sketch2Code Local")

st.title("🎨 Sketch2Code (Local AI)")
st.markdown("""
Transformez vos croquis à main levée en code HTML/Tailwind fonctionnel 
en utilisant **Ollama** et **LLaVA** localement.
""")

# Sidebar pour les réglages
with st.sidebar:
    st.header("Paramètres")
    model_choice = st.selectbox(
        "Modèle Vision (doit être installé via Ollama)", 
        ["llava", "moondream", "bakllava"], 
        index=0
    )
    st.info(f"Assurez-vous d'avoir lancé `ollama pull {model_choice}` dans votre terminal.")
    
    tech_stack = st.selectbox(
        "Stack Technique",
        ["HTML + Tailwind CSS", "HTML + CSS Classique", "React + Tailwind"]
    )

# Fonction pour convertir l'image en bytes pour Ollama
def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    return img_byte_arr.getvalue()

# Le Prompt système pour guider l'IA
def get_prompt(stack):
    return f"""
    You are an expert web developer. 
    Analyze the provided hand-drawn sketch of a user interface.
    Write the full working code to implement this interface.
    
    Target Framework: {stack}
    
    Rules:
    1. Analyze the layout, text, inputs, and buttons in the drawing.
    2. Use placeholder images (like https://placehold.co/600x400) if needed.
    3. Make it responsive and modern.
    4. Return ONLY the code inside markdown code blocks. Do not add explanations.
    5. If the handwriting is hard to read, infer the most logical context.
    """

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Télécharger votre croquis")
    uploaded_file = st.file_uploader("Choisissez une image (PNG, JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Votre croquis", use_container_width=True)
        
        generate_btn = st.button("✨ Générer le Code", type="primary")

# Variable pour stocker le code généré dans la session
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if uploaded_file is not None and generate_btn:
    with col2:
        with st.spinner(f"Analyse du dessin avec {model_choice} (cela peut prendre du temps selon votre GPU)..."):
            try:
                # Préparation de l'image
                img_bytes = image_to_bytes(image)
                
                # Appel à Ollama (Local)
                response = ollama.chat(
                    model=model_choice,
                    messages=[
                        {
                            'role': 'user',
                            'content': get_prompt(tech_stack),
                            'images': [img_bytes]
                        }
                    ]
                )
                
                # Extraction du contenu
                content = response['message']['content']
                st.session_state.generated_code = content
                
            except Exception as e:
                st.error(f"Erreur de connexion avec Ollama : {e}")
                st.warning("Assurez-vous que l'application Ollama tourne en arrière-plan.")

# Affichage du résultat
if st.session_state.generated_code:
    with col2:
        st.subheader("2. Code Généré")
        
        # Nettoyage basique du code (retirer les balises markdown ```html ... ```)
        clean_code = st.session_state.generated_code.replace("```html", "").replace("```css", "").replace("```", "")
        
        tab1, tab2 = st.tabs(["Aperçu (Rendu)", "Code Source"])
        
        with tab1:
            st.caption("Rendu approximatif du HTML")
            st.components.v1.html(clean_code, height=600, scrolling=True)
            
        with tab2:
            st.code(st.session_state.generated_code, language='html')
            st.download_button(
                label="Télécharger le fichier HTML",
                data=clean_code,
                file_name="sketch2code.html",
                mime="text/html"
            )