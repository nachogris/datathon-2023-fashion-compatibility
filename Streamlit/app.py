import pandas as pd
import streamlit as st

st.title("MANGO Outfit Generator")

clean_data = pd.read_csv("../datathon/dataset/clean_data.csv")

sel_prod_type = st.sidebar.selectbox("Select a product type", ['Bottoms', 'Dresses, jumpsuits and Complete set', 'Tops', 'Accessories, Swim and Intimate', 'Outwear'])

products = pd.read_csv("../datathon/dataset/product_data.csv")

subFrame = products[products['des_product_category'] == sel_prod_type].sample(frac = 1)

photos_to_show = subFrame[:5][['des_filename', 'cod_modelo_color']]
if 'photos_to_show' not in st.session_state:
      st.session_state.photos_to_show = photos_to_show


st.markdown("Select a product type in the sidebar menu. Some products will be displayed. If another sample of products is wanted, click the botton below.")

regenerate = st.button("Regenerate product sample")

if regenerate:
    st.session_state.photos_to_show = subFrame[:5][['des_filename', 'cod_modelo_color']]

st.markdown("## Select a seed product. This will be the main product of your outfit.")

col = st.columns(5)

for i, image_path in enumerate(st.session_state.photos_to_show['des_filename']):       
        image_path = "../" + image_path
        col[i].image(image_path)

st.session_state.seed_product = st.selectbox("Select the image corresponding to the wanted seed product", ['Image 1', 'Image 2', 'Image 3', 'Image 4', 'Image 5'])

transfor_prod = {
      "Image 1": st.session_state.photos_to_show['cod_modelo_color'].iloc[0],
      "Image 2": st.session_state.photos_to_show['cod_modelo_color'].iloc[1],
      "Image 3": st.session_state.photos_to_show['cod_modelo_color'].iloc[2],
      "Image 4": st.session_state.photos_to_show['cod_modelo_color'].iloc[3],
      "Image 5": st.session_state.photos_to_show['cod_modelo_color'].iloc[4],
}

# Read products data
products = pd.read_csv("../datathon/dataset/product_data.csv")

