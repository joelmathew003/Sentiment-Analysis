import streamlit as st 
import numpy as np
import keras
from keras.models import load_model
import librosa


st.title("Sentiment Analysis from Audio")
st.subheader("Aim: To give a prediction of the mood based on the audio")
audio = st.file_uploader("Please upload your audio in .wav format", type=["wav"])
if audio is not None:
    #st.write(type(audio))
    #st.write(dir(audio))
    details = {"name":audio.name,"dir":audio.__dir__,"type":audio.type}
    #st.write(details)
    signal, sample_rate = librosa.load(audio)
    mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = np.asarray([np.asarray(mfcc.T)])
    #st.write(mfcc.shape)
    loaded_model = load_model("/Users/Joel Mathew/Downloads/sentiment_analysis_model_v1.h5")

    predicted_output = np.argmax(loaded_model.predict(mfcc))
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