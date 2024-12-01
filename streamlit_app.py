import streamlit as st
import whisper
import io
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the Whisper model (choose small, medium, or large depending on your needs)
model = whisper.load_model("base")  # You can use "small", "medium", or "large" models for better accuracy

# Streamlit App UI
st.title("Customer Support Call Analysis")
st.write("Record and analyze customer support calls. Get transcription and feedback analysis.")

# Audio file upload for customer support call
audio_file = st.file_uploader("Upload an audio file of the customer support call", type=["wav", "mp3", "flac"])

# Debugging: Print file details
if audio_file is not None:
    st.write(f"Audio File Name: {audio_file.name}")
    st.write(f"File Size: {len(audio_file.getvalue())} bytes")

    st.audio(audio_file, format="audio/wav")

    # Function to transcribe the audio using Whisper
    def transcribe_audio(file):
        # Convert audio file to bytes
        audio_bytes = file.read()

        # Save the file temporarily for Whisper processing
        with open("temp_audio_file", "wb") as temp_file:
            temp_file.write(audio_bytes)

        try:
            # Use Whisper to transcribe the audio file
            result = model.transcribe("temp_audio_file")

            # Get the transcription text
            transcription = result["text"]
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
                # Perform a simple sentiment analysis (you can replace this with a more sophisticated analysis)
                feedback = st.session_state.transcription
                sentiment = "Positive" if "good" in feedback.lower() else "Negative"
                
                # Display sentiment analysis
                st.write("Feedback Sentiment Analysis:")
                st.write(f"Sentiment: {sentiment}")
            except Exception as e:
                st.error(f"Error during feedback analysis: {e}")
                logging.error(f"Error during feedback analysis: {e}")
        else:
            st.warning("Please transcribe a call first.")

else:
    st.info("Upload a customer support call audio file to start.")
