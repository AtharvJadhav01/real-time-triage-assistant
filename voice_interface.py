"""
Voice interface for speech-to-text and text-to-speech.
"""

import pyttsx3
from faster_whisper import WhisperModel


class VoiceInterface:

    def __init__(self):
        self.tts = pyttsx3.init()

        # Fast Whisper model
        self.stt = WhisperModel("base", compute_type="int8")

    def speech_to_text(self, audio_path):

        segments, _ = self.stt.transcribe(audio_path)

        text = ""
        for seg in segments:
            text += seg.text

        return text

    def text_to_speech(self, text):

        self.tts.say(text)
        self.tts.runAndWait()