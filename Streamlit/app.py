import pandas as pd
import streamlit as st
import tensorflow
import pickle
from tensorflow.keras.models import load_model

st.title("MANGO Outfit Generator")


sel_prod_type = st.sidebar.selectbox("Select a product type", ['Bottoms', 'Dresses, jumpsuits and Complete set', 'Tops', 'Accesories, Swim and Intimate', 'Outerwear'])

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

#@st.cache(allow_output_mutation=True)  # Set allow_output_mutation=True to handle mutable objects like models
def load_trained_model():
    # Replace 'trained_model.h5' with the actual path to your saved model
    model_path = '../datathon/trained_model.h5'
    loaded_model = load_model(model_path)
    return loaded_model

st.session_state.prod_test = products[products['cod_modelo_color'] == transfor_prod[st.session_state.seed_product]].squeeze()

model = load_trained_model()

def get_test(prod_test):
      import random

      des_product_category = ['Bottoms', 'Dresses, jumpsuits and Complete set', 'Tops','Accesories, Swim and Intimate', 'Outerwear']


      # Matrix with accepted outfit configurations per given seed product
      m_des_product_category = [ [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
      ]

      # Only able to select or dress or bottoms + tops. Choose this option randomly
      if random.choice([True, False]):
            m_des_product_category[3][0:3] = [0,1,0]
      if random.choice([True, False]):
            m_des_product_category[4][0:3] = [0,1,0]

      # Dicts to check dimensions of matrix and data in the dataset
      ind2 = {value: index for index, value in enumerate(des_product_category)}
      ind3 = {index: value for index, value in enumerate(des_product_category)}

      # Colors for the data model that will predict matchings
      columns = ['ColorSpecification1', 'Color1', 'Fabric1', 'Category1', 'AggregatedFamily1', 'Family1', 'Type1', 'ColorSpecification2', 'Color2', 'Fabric2', 'Category2', 'AggregatedFamily2', 'Family2', 'Type2']
      data_model = pd.DataFrame(columns=columns)

      # Colors to get info from the products dataset
      cols_clean_data = ['des_color_specification_esp', 'des_agrup_color_eng', 'des_fabric', 'des_product_category', 'des_product_aggregated_family', 'des_product_family', 'des_product_type']

      # Merge two rows to return the row that fits the predicting dataset
      def compute_row(prod1, prod2):
            prod1_list = []
            prod2_list = []
            for col in cols_clean_data:
                  prod1_list.append(prod1[col])
                  prod2_list.append(prod2[col])
                  
            return prod1_list + prod2_list  


      with open('../X_train_columns.pkl', 'rb') as f:
            X_train_columns = pickle.load(f)

      # Make outfit prediction
      outfit_prediction = []
      
      for index, col in enumerate(m_des_product_category[ind2[prod_test['des_product_category']]]): # recorre los 1s de la fila
            if col == 1: # accedim a la posició de la matriu
                  X_test = pd.DataFrame(columns = columns) # data frame de files st.session_state.prod_test amb totes els 'tops', per ex
                  X_test_index = []
                  for index, prod in products[products['des_product_category'] == ind3[index]].iterrows():
                        X_test.loc[len(X_test)] = compute_row(prod_test, prod)
                        X_test_index.append(prod['des_filename'])
                  
                  X_test = pd.get_dummies(X_test, columns=['ColorSpecification1', 'Color1', 'Fabric1', 'Category1', 'AggregatedFamily1', 'Family1', 'Type1', 'ColorSpecification2', 'Color2', 'Fabric2', 'Category2', 'AggregatedFamily2', 'Family2', 'Type2'], prefix=['ColorSpecification1', 'Color1', 'Fabric1', 'Category1', 'AggregatedFamily1', 'Family1', 'Type1', 'ColorSpecification2', 'Color2', 'Fabric2', 'Category2', 'AggregatedFamily2', 'Family2', 'Type2'])
                  missing_columns = set(X_train_columns) - set(X_test.columns)
                  
                  # Add missing columns to X_test
                  for col in missing_columns:
                        X_test[col] = 0

                  # Reorder columns in X_test
                  X_test = X_test[X_train_columns]
                  # Separate features and target variable in X_test
                  #X_test = X_test.drop(['Match'], axis=1)
                  # Identify boolean columns
                  bool_columns_test = X_test.select_dtypes(include='bool').columns
                  # Convert False to 0 and True to 1 for boolean columns
                  X_test[bool_columns_test] = X_test[bool_columns_test].astype(int)

                  prediction = model.predict(X_test)
                  
                  
                  result = [(X_test_index[i], prediction[i]) for i in range(len(prediction))]
                  sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
                  
                  outfit_prediction.append(sorted_result[0])
                  
      return outfit_prediction

outfit_prediction = get_test(st.session_state.prod_test)
outfit_prediction.append([st.session_state.prod_test['des_filename'],0])

st.markdown("### Suggested Outfit:")
col2 = st.columns(len(outfit_prediction))
for i, image_path in enumerate(outfit_prediction):       
        image_path = "../" + image_path[0]
        col2[i].image(image_path)
