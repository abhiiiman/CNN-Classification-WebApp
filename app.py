import streamlit as st
import tensorflow as tf
from keras.preprocessing import image # type: ignore
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# setting up the page header here.
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

# setting up the page config here.
st.set_page_config(
    page_title="DogCat CNN Classifier",
    page_icon="😽",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## Basic CNN App to classfiy Dog or Cat"
    }
)

# removing all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model(r'model.h5')

# Function to make a prediction
def make_prediction(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    result = model.predict(img_array)
    return 'dog' if result[0][0] > 0.5 else 'cat'

# Streamlit app
st.title("Cat or Dog Classifier 🐕🐈")
st.markdown("##### Upload an image and the model will predict whether it's a cat or a dog.")
st.markdown("* `92% Accuracy`⚡")

uploaded_file = st.file_uploader("Choose an image 🖼️", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    prediction = make_prediction(uploaded_file)
    if prediction == 'dog':
        st.markdown("### I can see, it's a Dog 🐶")
    else:
        st.markdown("### Okay so, it's a Cat 😽")
