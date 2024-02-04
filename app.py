import streamlit as st 
import numpy as np
import keras
from keras.models import load_model
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt

def audio_feature_plot_mfcc(plot_name):
    for audio_path in os.listdir("tempDir"):
        audio_path = os.path.join("tempDir",audio_path)
        (xf, sr) = librosa.load(audio_path)    

        if plot_name == "MFCC":

            mfccs = librosa.feature.mfcc(y=xf, sr=sr, n_mfcc=4)
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
            plt.title('MFCC Spectogram')
            plt.savefig("audio_features_mfcc")
            st.markdown("<h3 style='text-align: center; color: yellow;'> The MFCC features graph of the audio file uploaded is shown below</h1>",  unsafe_allow_html=True)
            st.image("audio_features_mfcc.png")

        elif plot_name == "Chroma":

            chroma = librosa.feature.chroma_stft(y=xf,sr = sr)
            librosa.display.specshow(chroma,x_axis='time')
            plt.colorbar()
            plt.title('Chroma Spectogram')
            plt.savefig("audio_features_chroma")
            st.markdown("<h3 style='text-align: center; color: yellow;'> The Chroma features graph of the audio file uploaded is shown below</h1>",  unsafe_allow_html=True)
            st.image("audio_features_chroma.png")

        elif plot_name == "Mel Scale":

            mel = librosa.feature.melspectrogram(y=xf,sr=sr)
            librosa.display.specshow(mel,x_axis="time")
            plt.colorbar()
            plt.title("Mel Spectogram")
            plt.savefig("audio_features_mel")
            st.markdown("<h3 style='text-align: center; color: yellow;'> The Mel features graph of the audio file uploaded is shown below</h1>",  unsafe_allow_html=True)
            st.image("audio_features_mel.png")

def model(audio):
    signal, sample_rate = librosa.load(audio)
    mfcc = librosa.feature.mfcc(signal=signal, sample_rate=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
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

st.markdown("<h1 style='text-align: center; color: yellow;'>Speech Emotion Recognition</h1>",  unsafe_allow_html=True)
st.sidebar.subheader("About")
st.sidebar.info("Aim: To give a prediction of the mood of the speaker based on the audio")
     

def main():
    audio = st.file_uploader("Please upload your audio here", type=["wav","mp3"])   
    if audio is not None:
        with open(os.path.join("tempDir",audio.name),"wb") as f:
            f.write(audio.getbuffer())

        st.sidebar.title('Select the spectrogram to view')
        plot_name = st.sidebar.radio("",("MFCC","Chroma","Mel Scale"))

        audio_feature_plot_mfcc(plot_name)

        model(audio)


def cache():
    number_of_files = 0
    for f in os.listdir("tempDir"):
        if f:
            number_of_files += 1

    if number_of_files == 0:
        main()
    else:
        for f in os.listdir("tempDir"):
            if f:
                audio_path = os.path.join("tempDir",f)
                os.remove(audio_path)
        main()

cache()