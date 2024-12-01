import streamlit as st
import whisper
import io
import logging
from pydub import AudioSegment

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load the Whisper model (choose small, medium, or large depending on your needs)
model = whisper.load_model("base")  # You can use "small", "medium", or "large" models for better accuracy

# Streamlit App UI
st.title("Customer Support Call Analysis")
st.write("Record and analyze customer support calls. Get transcription, feedback analysis, and sentiment summary.")

# Audio file upload for customer support call
audio_file = st.file_uploader("Upload an audio file of the customer support call", type=["wav", "mp3", "flac"])

# Debugging: Print file details
if audio_file is not None:
    st.write(f"Audio File Name: {audio_file.name}")
    st.write(f"File Size: {len(audio_file.getvalue())} bytes")
    st.audio(audio_file, format="audio/wav")

    # Function to convert audio to WAV format using pydub (without ffmpeg)
    def convert_audio_to_wav(file):
        audio = AudioSegment.from_file(file)
        wav_file_path = "converted_audio.wav"
        audio.export(wav_file_path, format="wav")
        return wav_file_path

    # Function to transcribe the audio using Whisper
    def transcribe_audio(file):
        # Convert audio file to WAV format
        wav_file = convert_audio_to_wav(file)

        try:
            # Use Whisper to transcribe the audio file
            result = model.transcribe(wav_file)

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

    # Analyze the feedback for sentiment and summary using Gemini AI
    if st.button("Analyze Feedback and Generate Summary"):
        if 'transcription' in st.session_state and st.session_state.transcription:
            try:
                # Use Gemini to analyze sentiment of the transcription (feedback analysis)
                feedback = st.session_state.transcription
                model = genai.GenerativeModel("gemini-1.5-flash")  # You can replace with a more suitable model

                # Generate sentiment analysis
                prompt = f"Analyze the sentiment of the following customer support feedback:\n\n{feedback}"
                sentiment_response = model.generate_content(prompt)

                # Get sentiment analysis response
                sentiment = sentiment_response.text.strip()
                st.write("Feedback Sentiment Analysis:")
                st.write(f"Sentiment: {sentiment}")

                # Generate summary of the transcription
                summary_prompt = f"Summarize the following customer support call transcription:\n\n{feedback}"
                summary_response = model.generate_content(summary_prompt)

                # Get summary of the call
                summary = summary_response.text.strip()
                st.write("Generated Summary:")
                st.write(summary)

            except Exception as e:
                st.error(f"Error during feedback analysis or summary generation: {e}")
                logging.error(f"Error during feedback analysis or summary generation: {e}")
        else:
            st.warning("Please transcribe a call first.")

else:
    st.info("Upload a customer support call audio file to start.")
