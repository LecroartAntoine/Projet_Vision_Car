import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import os
from glob import glob

# Importer uniquement ce dont nous avons besoin depuis notre fichier utilitaire simplifié
from utils_viz import CATEGORY_INFO, mask_to_rgb

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Future Vision | Démo", page_icon="🚗", layout="wide")

# --- CONFIGURATION DE L'API ET DES DONNÉES ---
API_URL = "http://127.0.0.1:8000/predict"
# Le dossier de test contient uniquement les images, pas les masques.
TEST_IMAGE_DIR = "data/leftImg8bit/test" 

# --- FONCTIONS HELPER ---
@st.cache_data
def get_test_image_files():
    """Charge la liste des images de test une seule fois."""
    if not os.path.exists(TEST_IMAGE_DIR):
        return []
    # On cherche récursivement les images dans les sous-dossiers des villes
    image_files = glob(os.path.join(TEST_IMAGE_DIR, "*", "*.png"))
    return [os.path.basename(f) for f in image_files]

# --- INTERFACE UTILISATEUR ---
st.title("🚗 Démonstration de l'API de Segmentation")

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header("Contrôles")
    source_choice = st.radio(
        "Choisissez la source de l'image",
        ("Téléverser une image", "Sélectionner une image de test"),
        horizontal=True,
    )
    
    image_file = None
    if source_choice == "Téléverser une image":
        uploaded_file = st.file_uploader("Choisissez un fichier image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_file = uploaded_file
    else:
        test_image_list = get_test_image_files()
        if not test_image_list:
            st.error(f"Le dossier d'images de test est vide ou n'a pas été trouvé à l'emplacement : `{TEST_IMAGE_DIR}`")
        else:
            selected_image_name = st.selectbox("Sélectionnez une image de test", test_image_list)
            if selected_image_name:
                # Reconstruire le chemin complet de l'image sélectionnée
                city_folder = selected_image_name.split('_')[0]
                image_file = os.path.join(TEST_IMAGE_DIR, city_folder, selected_image_name)

    st.header("Légende des Couleurs")
    for cat_id, info in CATEGORY_INFO.items():
        color_hex = '#%02x%02x%02x' % info['color']
        st.markdown(
            f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
            f"<div style='width: 20px; height: 20px; background-color: {color_hex}; border: 1px solid #444; margin-right: 10px;'></div>"
            f"<span>{info['name'].capitalize()}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# --- AFFICHAGE PRINCIPAL ---
if image_file:
    # Ouvrir l'image
    original_image = Image.open(image_file).convert("RGB")
    
    # La disposition est toujours sur deux colonnes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Originale")
        st.image(original_image, use_container_width=True)

    if st.button("Lancer la prédiction", use_container_width=True, type="primary"):
        with st.spinner("Prédiction en cours... L'API travaille..."):
            try:
                # Préparer l'image pour l'envoi à l'API
                img_byte_arr = io.BytesIO()
                original_image.save(img_byte_arr, format='PNG')
                files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                
                # Appeler l'API
                response = requests.post(API_URL, files=files, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_mask_2d = np.array(result['predicted_mask'], dtype=np.uint8)
                    predicted_mask_rgb = mask_to_rgb(predicted_mask_2d)
                    
                    # Afficher le masque prédit dans la deuxième colonne
                    with col2:
                        st.subheader("Masque Prédit par l'API")
                        st.image(predicted_mask_rgb, caption="Segmentation retournée par l'API.", use_container_width=True)
                else:
                    st.error(f"Erreur de l'API (Code: {response.status_code}):")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.error(f"Erreur de connexion à l'API : {e}")

else:
    st.info("⬅️ Veuillez choisir une source d'image dans la barre latérale pour commencer.")