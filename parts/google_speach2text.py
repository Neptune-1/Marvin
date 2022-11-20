from google.cloud import speech


class Speech2Text:
    def __init__(self):
        # Instantiates a client
        self.client = speech.SpeechClient()

    def transcribe_speech(self, data):
        audio = speech.RecognitionAudio(audio_content=data)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        # Detects speech in the audio file
        response = self.client.recognize(config=config, audio=audio)

        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
