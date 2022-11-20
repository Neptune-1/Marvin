import os
import uuid
from time import time

import pyaudio
import numpy as np
from scipy.io import wavfile
from parts.lamp import Lamp
from parts.model import Model


# from google_speach2text import Speech2Text


class Observer:
    def __init__(self):
        self.RATE = 16000
        self.CHUNKS_PER_SECOND = 2
        self.FRAMES_PER_BUFFER = 16000 // self.CHUNKS_PER_SECOND
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1

        self.buffer = b""
        self.chunk_number = 0
        self.START_TIME = time()

        self.audio = None
        self.stream = None

        self.model = Model()

        self.lamp = Lamp()

        # self.speech2text = Speech2Text()

    def start_stream(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=1
        )

    def stop(self):
        self.stream.close()
        self.audio.terminate()

    def observe(self):
        self.start_stream()
        while True:  # for _ in range(int(self.RATE / self.FRAMES_PER_BUFFER * 5))
            data = self.stream.read(self.FRAMES_PER_BUFFER, exception_on_overflow=False)
            if self.chunk_number % self.CHUNKS_PER_SECOND == 0 and self.chunk_number != 0:
                self.check_for_catch_phrase(self.buffer)
                self.buffer = b""

            self.chunk_number += 1

            self.buffer += data
        
    def save2wav(self, data):
        dir = os.path.join("new_marvins", f"{uuid.uuid4()}.wav")
        wavfile.write(dir, self.RATE, data)
        return dir

    def check_for_catch_phrase(self, data):
        data_array = (np.fromstring(data, dtype=np.float32))
        output = self.model.predict(data_array.reshape((1, -1)), self.RATE)
        res = np.argmax(output[0])
        print(res)
        if res == 1:
            self.save2wav(data_array)
            self.react_to_catch_phrase()
            # self.speech2text.transcribe_speech(data)

    def react_to_catch_phrase(self):
        self.lamp.toggle()
