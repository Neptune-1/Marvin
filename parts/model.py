import librosa
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path="models/model-1.tflite")
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_size = self.input_details[0]['shape'][1]
        self.sample_rate = 44100

    def predict(self, data, rate):
        if rate != self.sample_rate:
            data = librosa.resample(data, orig_sr=rate, target_sr=self.sample_rate)

        if len(data) < self.input_size:
            data.resize(self.input_size)
        data = np.expand_dims(data[:self.input_size], axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
