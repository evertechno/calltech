import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import logging
import google.generativeai as genai
import time

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Set up Gemini AI (ensure API key is set in Streamlit secrets)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Customer Support Call Analysis")
st.write("Record and analyze customer support calls. Get transcription and feedback analysis.")

# Limit file size to 200MB and allow multiple formats
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# Audio file upload for customer support call (allowing WAV, MP3, FLAC, AIFF, MP4, and MP4A)
audio_file = st.file_uploader("Upload an audio file of the customer support call", 
                              type=["wav", "mp3", "flac", "aiff", "mp4", "mp4a"])

# Ensure that only audio files are allowed and check for file size
if audio_file is not None:
    # Check if the file is too large
    if len(audio_file.getvalue()) > MAX_FILE_SIZE:
        st.error("The file is too large. Please upload a file smaller than 200MB.")
    else:
        st.write(f"Audio File Name: {audio_file.name}")
        st.write(f"File Size: {len(audio_file.getvalue()) / (1024 * 1024):.2f} MB")
        
        st.audio(audio_file, format="audio/wav")

        # Function to convert MP4A (or other formats) to WAV using pydub
        def convert_to_wav(file):
            try:
                # Check file extension
                file_extension = file.name.split('.')[-1].lower()
                st.write(f"File extension detected: {file_extension}")

                # Only proceed if the file is mp4a or mp4
                if file_extension in ['mp4a', 'mp4']:
                    st.write(f"Converting {file_extension} to WAV...")
                    # Load the uploaded file using pydub (it can handle .mp4a files as well)
                    audio = AudioSegment.from_file(file)

                    # Export the audio as WAV format
                    wav_file = io.BytesIO()
                    audio.export(wav_file, format="wav")
                    wav_file.seek(0)
                    st.write("File has been converted to valid WAV format.")
                    return wav_file
                else:
                    st.error(f"Unsupported file extension: {file_extension}")
                    return None
            except Exception as e:
                st.error(f"Error converting file to WAV: {e}")
                return None

        # Function to transcribe the audio using SpeechRecognition
        def transcribe_audio(file):
            recognizer = sr.Recognizer()

            # Convert the uploaded audio file to WAV if it's not already WAV
            if file.type != "audio/wav":
                file = convert_to_wav(file)
                if not file:
                    return None

            # Convert the WAV file into an audio source
            with io.BytesIO(file.read()) as audio_file_io:
                with sr.AudioFile(audio_file_io) as source:
                    audio = recognizer.record(source)

            retries = 3
            for i in range(retries):
                try:
                    # Try to use Google Web Speech API to transcribe audio
                    transcription = recognizer.recognize_google(audio)
                    return transcription
                except sr.UnknownValueError:
                    st.error("Google Speech Recognition could not understand the audio")
                    return None
                except sr.RequestError as e:
                    if i < retries - 1:
                        time.sleep(2 ** i)  # Exponential backoff
                    else:
                        st.error(f"Could not request results from Google Speech Recognition service; {e}")
                        return None
            return None

        # Fallback to Sphinx if Google API fails
        def transcribe_with_fallback(file):
            transcription = transcribe_audio(file)
            if transcription is None:
                st.warning("Google Speech Recognition failed. Trying Sphinx (local)...")
                recognizer = sr.Recognizer()
                with io.BytesIO(file.read()) as audio_file_io:
                    with sr.AudioFile(audio_file_io) as source:
                        audio = recognizer.record(source)
                try:
                    transcription = recognizer.recognize_sphinx(audio)
                    return transcription
                except sr.UnknownValueError:
                    st.error("Sphinx could not understand the audio")
                except sr.RequestError as e:
                    st.error(f"Sphinx request failed; {e}")
            return transcription

        # Transcribe the uploaded audio
        if st.button("Transcribe Call"):
            with st.spinner("Transcribing..."):
                transcription = transcribe_with_fallback(audio_file)
                if transcription:
                    st.write("Transcription:")
                    st.write(transcription)
                    # Save the transcription to session state for later use
                    st.session_state.transcription = transcription
                else:
                    st.warning("Could not transcribe the audio.")

        # Analyze the feedback for sentiment using Gemini AI (feedback analysis)
        if st.button("Analyze Feedback"):
            if 'transcription' in st.session_state and st.session_state.transcription:
                try:
                    # Use Gemini to analyze sentiment of the transcription (feedback analysis)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"Analyze the following customer support feedback and provide a sentiment analysis:\n\n{st.session_state.transcription}"
                    
                    # Call Gemini AI to analyze feedback
                    response = model.generate_content(prompt)
                    
                    # Display sentiment analysis
                    st.write("Feedback Sentiment Analysis:")
                    st.write(f"Sentiment: {response.text}")
                except Exception as e:
                    st.error(f"Error during feedback analysis: {e}")
            else:
                st.warning("Please transcribe a call first.")
else:
    st.info("Upload a customer support call audio file to start.")
