import streamlit as st
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)
import numpy as np
from PIL import Image


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from disease_info import disease_data

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# ---------------------------------
# UI TEXT
# ---------------------------------

ui_text = {

    "english": {
        "title": "🌿 Smart Leaf Disease Detection System",
        "disease": "🦠 Disease Detected",
        "solution": "💡 Solution",
        "fertilizer": "🌱 Recommended Fertilizer",
        "confidence": "Prediction Confidence",
        "upload": "📤 Upload Leaf Image",
        "camera": "📸 Take Leaf Photo"
    },

    "hindi": {
        "title": "🌿 स्मार्ट पत्ती रोग पहचान प्रणाली",
        "disease": "🦠 रोग की पहचान",
        "solution": "💡 समाधान",
        "fertilizer": "🌱 अनुशंसित उर्वरक",
        "confidence": "भविष्यवाणी सटीकता",
        "upload": "📤 पत्ती की फोटो अपलोड करें",
        "camera": "📸 पत्ती की फोटो लें"
    },

    "marathi": {
        "title": "🌿 स्मार्ट पान रोग ओळख प्रणाली",
        "disease": "🦠 रोग ओळख",
        "solution": "💡 उपाय",
        "fertilizer": "🌱 शिफारस केलेले खत",
        "confidence": "अचूकता",
        "upload": "📤 पानाचा फोटो अपलोड करा",
        "camera": "📸 पानाचा फोटो काढा"
    }

}

# ---------------------------------
# LOAD MODEL
# ---------------------------------



model = load_model(
    "model/final_leaf_model.keras",
    compile=False
)


# ---------------------------------
# CLASS NAMES
# ---------------------------------

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ---------------------------------
# LANGUAGE
# ---------------------------------

language = st.selectbox(
    "🌍 Select Language / भाषा / भाषा निवडा",
    ["english", "hindi", "marathi"]
)

text = ui_text[language]

# ---------------------------------
# TITLE
# ---------------------------------

st.title(text["title"])

# ---------------------------------
# IMAGE INPUT
# ---------------------------------

uploaded_file = st.file_uploader(
    text["upload"],
    type=["jpg", "jpeg", "png"]
)

camera_photo = st.camera_input(
    text["camera"]
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")

# ---------------------------------
# PREDICTION
# ---------------------------------

if image is not None:

    st.image(
        image,
        caption="Leaf Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    predicted_class = class_names[predicted_index]

    st.write("Predicted Class:", predicted_class)
    st.write("Confidence:", confidence)

    if predicted_class in disease_data:

        info = disease_data[predicted_class][language]

        st.subheader(text["disease"])
        st.success(info["disease"])

        st.metric(
            label=text["confidence"],
            value=f"{confidence:.2f}%"
        )

        st.subheader(text["solution"])
        st.info(info["solution"])

        st.subheader(text["fertilizer"])
        st.warning(info["fertilizer"])

        st.subheader("📋 Technical Prediction")
        st.code(predicted_class)
