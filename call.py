import base64
import requests
from pydub import AudioSegment
import io

def mp3_to_wav_bytes(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

def save_history_base64(history_base64, filename):
    with open(filename, 'w') as f:
        f.write(history_base64)

# Load the history as base64 encoded string
with open('updated_history.pickle', 'r') as f:
    history_base64 = f.read()

# Path to your MP3 file
mp3_file_path = "Hello.mp3"

# Convert MP3 to WAV bytes
wav_bytes = mp3_to_wav_bytes(mp3_file_path)

# Prepare the API call
url = "http://127.0.0.1:8000/talk_to_agents/"
files = {
    'audio_file': ('audio.wav', wav_bytes, 'audio/wav'),
    'history_file': ('history.pickle', history_base64, 'application/octet-stream')
}
data = {
    'agent_number': 4,  # Example agent number
    'gemini_api_key': "***************"
}

# Make the API call
response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    narration = result['narration']
    updated_history_base64 = result['updated_history']
    status = result['status']
    print(f"Narration: {narration}")
    print(f"Status: {status}")

    # Save the updated history
    save_history_base64(updated_history_base64, 'updated_history.pickle')
    print("Updated history saved to: updated_history.pickle")
else:
    print(f"Error: {response.status_code} - {response.text}")