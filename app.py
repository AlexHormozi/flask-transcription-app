from flask import Flask, request, jsonify
import numpy as np
from faster_whisper import WhisperModel

app = Flask(__name__)

# Load the Faster Whisper model (choose an appropriate model size)
model = WhisperModel("large-v2")  # Adjust based on your needs

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_data = request.data  # Get raw audio data from the request

    # Convert raw audio data to NumPy array (assuming 16-bit PCM format)
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Transcribe the audio using Faster Whisper
    segments, _ = model.transcribe(audio_np, beam_size=5, language="en", output_language="fr")  # Change "fr" to your desired language

    transcription = " ".join(segment.text for segment in segments)

    return jsonify({"transcription": transcription})

if __name__ == "__main__":
    app.run(debug=True)
