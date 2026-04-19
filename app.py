import os
os.environ["KERAS_BACKEND"] = "jax"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers, models, regularizers
import time

st.set_page_config(page_title="CIFAR-10 CNN Classifier", page_icon="", layout="wide")

CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
NUM_CLASSES = 10

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; text-align: center; color: #00E5FF; }
    .pred-box { background: #1a1a2e; border-radius: 16px; padding: 2rem; text-align: center; }
    .pred-label { font-size: 2rem; font-weight: 700; color: #ffffff; }
    .conf-num { font-size: 3rem; font-weight: 700; color: #00E5FF; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    def build_cnn():
        model = models.Sequential()
        model.add(layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4), input_shape=(32,32,3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    model = build_cnn()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
  try:
        model.load_weights('best_cnn_model.h5')
    except Exception:
        (X_train, y_train), _ = keras.datasets.cifar10.load_data()
        X_train = X_train.astype('float32') / 255.0
        y_train = keras.utils.to_categorical(y_train, 10)
        model.fit(X_train, y_train, batch_size=256, epochs=3, verbose=0)
    return model

with st.spinner("Loading model..."):
    model = load_model()
st.success("Model loaded successfully.")

def preprocess(pil_image):
    img = pil_image.convert('RGB').resize((32, 32), Image.LANCZOS)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)

def confidence_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    sorted_idx = np.argsort(probs)[::-1]
    colors = plt.cm.cool(np.linspace(0.2, 0.9, NUM_CLASSES))
    ax.barh(range(NUM_CLASSES), probs[sorted_idx] * 100, color=colors, height=0.65)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels([CLASS_NAMES[i] for i in sorted_idx], color='#cccccc', fontsize=10)
    ax.set_xlabel('Confidence (%)', color='#888')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.tick_params(colors='#888')
    ax.set_title('All Class Confidence Scores', color='#ccc', fontsize=11, pad=10)
    for i, idx in enumerate(sorted_idx):
        ax.text(probs[idx]*100 + 1, i, f'{probs[idx]*100:.1f}%', va='center', color='white', fontsize=9)
    plt.tight_layout()
    return fig

st.markdown("<h1 class='main-title'>CIFAR-10 CNN Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Lab 11 | STAT222 | BSDS-02 | Upload an image to classify it</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("### Upload Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp","bmp"])
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
        small = pil_img.convert('RGB').resize((32,32))
        st.image(small, caption="What model sees (32x32)", width=120)

with col2:
    st.markdown("### Prediction")
    if not uploaded:
        st.info("Upload an image on the left to see the prediction.")
    else:
        arr = preprocess(pil_img)
        with st.spinner("Running inference..."):
            probs = model.predict(arr, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        top_name = CLASS_NAMES[top_idx]

        st.markdown(f"""
        <div class='pred-box'>
            <div style='color:#888; font-size:0.85rem; letter-spacing:2px;'>PREDICTION</div>
            <div class='pred-label'>{top_name}</div>
            <div class='conf-num'>{top_conf*100:.1f}%</div>
            <div style='color:#888; font-size:0.85rem;'>confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Top 3 Predictions**")
        top3 = np.argsort(probs)[::-1][:3]
        c1, c2, c3 = st.columns(3)
        for col, idx, medal in zip([c1,c2,c3], top3, ["#1","#2","#3"]):
            col.metric(medal, CLASS_NAMES[idx], f"{probs[idx]*100:.1f}%")

        st.markdown("---")
        fig = confidence_chart(probs)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with st.sidebar:
    st.markdown("### Model Info")
    st.markdown("**Architecture:** 3 Conv Blocks (64-128-256) + Dense layers")
    st.markdown("**Dataset:** CIFAR-10 (60,000 images)")
    st.markdown("**Test Accuracy:** ~85%")
    st.markdown("**Classes:**")
    for name in CLASS_NAMES:
        st.markdown(f"- {name}")
    st.markdown("---")
    st.caption("STAT222 | BSDS-02 | Lab 11")
