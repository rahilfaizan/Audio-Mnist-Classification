import streamlit as st
from audio_recorder_streamlit import audio_recorder
import wave
from convert_audio import convert
from predict import predict
# Set the background image
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://images.unsplash.com/photo-1567360425618-1594206637d2?q=80&w=2160&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;  
#     background-repeat: no-repeat;
# }
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)



st.title('Antarctic Audio')
st.title('Audio-Mnist Classification')

audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0))


# Your desired font size
font_size = "40px"

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    audio_file = "audio.wav"
    with open(audio_file, "wb") as f:
        f.write(audio_bytes)
    prediction = predict(convert("audio.wav"))
    st.markdown(f"<p style='font-size: {font_size};'>Prediction: {prediction}</p>", unsafe_allow_html=True)
    st.image("spec.png", width=100)




