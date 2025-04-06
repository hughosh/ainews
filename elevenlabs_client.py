import os
import requests
from typing import Optional
import io
from pydub import AudioSegment

# Retrieve the ElevenLabs API key from the environment.
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")

def synthesize_tts(text: str, voice_id: str, stability: float = 0.5, similarity_boost: float = 0.5) -> bytes:
    """
    Synthesizes speech from text using the ElevenLabs API.
    
    Parameters:
      text (str): The text to convert to speech.
      voice_id (str): The voice ID to use.
      stability (float): Controls consistency of voice output (0.0 to 1.0).
      similarity_boost (float): Controls similarity to the voice sample (0.0 to 1.0).
    
    Returns:
      bytes: The synthesized speech audio as bytes (in mp3 format).
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "accept": "audio/mpeg"
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API request failed: {response.status_code} {response.text}")
    return response.content

def get_audio_length(audio_bytes: bytes) -> float:
    """
    Returns the length of the audio (in seconds) from the given mp3 data.
    
    Parameters:
      audio_bytes (bytes): The audio data in mp3 format.
    
    Returns:
      float: Duration in seconds.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    return len(audio) / 1000.0

def save_audio(audio_bytes: bytes, file_path: str) -> None:
    """
    Saves the audio bytes to a file.
    
    Parameters:
      audio_bytes (bytes): The audio data.
      file_path (str): Path where the audio file should be saved.
    """
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

