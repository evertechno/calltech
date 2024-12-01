import streamlit as st
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech
import io
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Configure the API key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set up Google Cloud Speech-to-Text client
client = speech.SpeechClient()

# Streamlit App UI
st.title("Customer Support Call Analysis")
st.write("Record and analyze customer support calls. Get transcription and feedback analysis.")

# Audio file upload for customer support call
audio_file = st.file_uploader("Upload an audio file of the customer support call", type=["wav", "mp3"])

# Debugging: Print file details
if audio_file is not None:
    st.write(f"Audio File Name: {audio_file.name}")
    st.write(f"File Size: {len(audio_file.getvalue())} bytes")

    st.audio(audio_file, format="audio/wav")

    # Function to transcribe the audio using Google Speech-to-Text
    def transcribe_audio(file):
        # Convert audio file to bytes
        audio_bytes = file.read()

        # Prepare the audio input for Speech-to-Text
        audio = speech.RecognitionAudio(content=audio_bytes)
        
        # Attempt to determine the correct encoding based on the file type
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16  # Default encoding
        if file.name.endswith("mp3"):
            encoding = speech.RecognitionConfig.AudioEncoding.MP3
        elif file.name.endswith("wav"):
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16

        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=16000,  # Ensure this matches the uploaded file's sample rate
            language_code="en-US",
        )

        try:
            # Perform the transcription
            response = client.recognize(config=config, audio=audio)

            # Extract the transcription from the response
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript + "\n"

            return transcription
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            logging.error(f"Error during transcription: {e}")
            return None

    # Transcribe the uploaded audio
    if st.button("Transcribe Call"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_file)
            if transcription:
                st.write("Transcription:")
                st.write(transcription)
                # Save the transcription to session state for later use
                st.session_state.transcription = transcription
            else:
                st.warning("Could not transcribe the audio.")

    # Analyze the feedback for sentiment (feedback analysis)
    if st.button("Analyze Feedback"):
        if 'transcription' in st.session_state and st.session_state.transcription:
            try:
                # Use Gemini to analyze sentiment of the transcription (e.g., feedback analysis)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Analyze the following customer support feedback and provide a sentiment analysis:\n\n{st.session_state.transcription}"
                response = model.generate_content(prompt)
                
                # Display sentiment analysis
                st.write("Feedback Sentiment Analysis:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error during feedback analysis: {e}")
                logging.error(f"Error during feedback analysis: {e}")
        else:
            st.warning("Please transcribe a call first.")

else:
    st.info("Upload a customer support call audio file to start.")
