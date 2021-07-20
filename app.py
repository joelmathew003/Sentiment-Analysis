import streamlit as st 
import numpy as np
import keras
from keras.models import load_model
import librosa


st.title("Sentiment Analysis from Audio")
st.sidebar.subheader("About")
st.sidebar.info("Aim: To give a prediction of the mood of the speaker based on the audio")
audio = st.file_uploader("Please upload your audio in .wav format", type=["wav"])
if audio is not None:
    signal, sample_rate = librosa.load(audio)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = np.asarray([np.asarray(mfcc.T)])
    loaded_model = load_model("sentiment_analysis_model_v1.h5")
    prediction = loaded_model.predict(mfcc)
    predicted_output = np.argmax(prediction)
    if(predicted_output==0):
        st.subheader("Prediction : Disgust")
    elif (predicted_output==1):
        st.subheader("Prediction : Happy")
    elif (predicted_output==2):
        st.subheader("Prediction : Sad")
    elif (predicted_output==3):
        st.subheader("Prediction : Neutral")
    elif (predicted_output==4):
        st.subheader("Prediction : Fear")
    elif (predicted_output==5):
        st.subheader("Prediction : Angry")
    if st.checkbox("Show Probability of each emotion"):
        prob = {}
        prob["Disgust"] = prediction.T[0]
        prob["Happy"] = prediction.T[1]
        prob["Sad"] = prediction.T[2]
        prob["Neutral"] = prediction.T[3]
        prob["Fear"] = prediction.T[4]
        prob["Anger"] = prediction.T[5]
        st.bar_chart(prob)
        st.table(prob)
    