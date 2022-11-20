import numpy as np
import tensorflow as tf
from IPython.core.display_functions import display
from IPython.lib.display import Audio
from scipy.io.wavfile import read
from parts.model import Model
import soundfile as sf
import os
import matplotlib.pyplot as plt
def basic_test():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="../models/model-1.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print(input_details)

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


def test_on_wav():
    a = read("5454f9ff-387d-48ce-97e3-9700c59a6ee4.wav")
    data = np.array(a[1], dtype=float)
    print(data.shape)
    model = Model()
    output = model.predict(data.reshape((1, -1)).astype(np.float32), 16000)
    res = np.argmax(output[0])
    print(res)


def show_sample(audio_path):
  audio_data, sample_rate = sf.read(audio_path)
  class_name = os.path.basename(os.path.dirname(audio_path))
  print(f'Class: {class_name}')
  print(f'File: {audio_path}')
  print(f'Sample rate: {sample_rate}')
  print(f'Sample length: {len(audio_data)}')

  plt.title(class_name)
  plt.plot(audio_data)
  display(Audio(audio_data, rate=sample_rate))

test_on_wav()