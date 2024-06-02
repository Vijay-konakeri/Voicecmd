import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.embeddings import OpenAIEmbeddings
import os
import io
import whisper
import tempfile
from gtts import gTTS
import numpy as np
import requests
import json
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE")

# Initialize the OpenAI object
llm = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the OpenAI Embeddings object
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Initialize Whisper model
model = whisper.load_model("base")

def create_agent(data, llm):
    """Create a Pandas DataFrame agent."""
    return create_pandas_dataframe_agent(llm, data, verbose=False)

def read_pdf(file):
    """Read PDF file using PyPDFLoader"""
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    return loader.load()

def read_csv(file):
    """Read CSV file and return as DataFrame"""
    return pd.read_csv(file)

def process_file(uploaded_file):
    """Process uploaded file and create an agent"""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            documents = read_pdf(uploaded_file)
            text = "\n\n".join([doc.page_content for doc in documents])
            df = pd.DataFrame({"text": [text]})
            st.session_state.agent = create_agent(df, llm)
        elif uploaded_file.type == "text/csv":
            df = read_csv(uploaded_file)
            st.session_state.agent = create_agent(df, llm)

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say something!")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def text_to_speech(text, voice_id=DEFAULT_VOICE):
    url = "https://api.elevenlabs.io/v1/text-to-speech"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_id": voice_id
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        audio_content = response.content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_content)
            return f.name
    else:
        st.write("Error: ", response.status_code, response.text)
        return None

def get_eleven_labs_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["voices"]
    else:
        st.write("Error fetching voices: ", response.status_code, response.text)
        return None

# Streamlit app interface
st.sidebar.title("Upload your file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or PDF file", type=["csv", "pdf"])
if uploaded_file:
    process_file(uploaded_file)
    st.sidebar.success("File uploaded and processed successfully!")

# Voice selection
voices_list = get_eleven_labs_voices()
if voices_list:
    voice_options = {v["name"]: v["voice_id"] for v in voices_list}
    selected_voice_name = st.sidebar.selectbox("Choose a voice:", list(voice_options.keys()))
    selected_voice = voice_options[selected_voice_name]
else:
    selected_voice = DEFAULT_VOICE
    st.sidebar.warning("Using default voice due to error fetching voices.")

# Main conversation area
st.title("Chat with your Data (Human-Like Voice)")
if "agent" in st.session_state:
    st.write("Ask your questions directly using your voice:")

    # Start recording button
    if st.button("Start Recording"):
        # Get user's voice input
        user_input = record_audio()
        if user_input:
            st.write(f"Recognized Text: {user_input}")  # Display the recognized text
            # Get LLM's response
            response = st.session_state.agent.run(user_input)
            st.text_area("Bot's response:", response, height=50)

            # Convert response to speech
            speech_file = text_to_speech(response, voice_id=selected_voice)
            if speech_file:
                st.write(f"Listen to the bot's response:")
                st.audio(speech_file)
                os.unlink(speech_file)
            else:
                st.warning("Failed to generate speech. Using default TTS as a fallback.")
                fallback_speech = gTTS(text=response, lang='en')
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    fallback_speech.save(temp_file.name)
                    st.audio(temp_file.name)
                    os.unlink(temp_file.name)
        else:
            st.warning("No text recognized from audio input.")

else:
    st.write("Please upload a file to start the conversation.")
