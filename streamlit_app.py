import streamlit as st
import speech_recognition as sr
import io
import logging
import google.generativeai as genai

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Set up Gemini AI (ensure API key is set in Streamlit secrets)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Streamlit App UI
st.title("Customer Support Call Analysis")
st.write("Record and analyze customer support calls. Get transcription and feedback analysis.")

# Audio file upload for customer support call
audio_file = st.file_uploader("Upload an audio file of the customer support call", type=["wav"])

# Debugging: Print file details
if audio_file is not None:
    st.write(f"Audio File Name: {audio_file.name}")
    st.write(f"File Size: {len(audio_file.getvalue())} bytes")
    
    st.audio(audio_file, format="audio/wav")

    # Function to transcribe the audio using SpeechRecognition
    def transcribe_audio(file):
        recognizer = sr.Recognizer()

        # Convert the WAV file into an audio source
        with io.BytesIO(file.read()) as audio_file_io:
            with sr.AudioFile(audio_file_io) as source:
                audio = recognizer.record(source)

        try:
            # Use Google Web Speech API to transcribe audio
            transcription = recognizer.recognize_google(audio)
            return transcription
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
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
