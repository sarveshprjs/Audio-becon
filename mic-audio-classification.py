import pyaudio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import urllib.request
import os

# ----------- Load YAMNet Model -----------
print("📦 Loading YAMNet model...")
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)
print("✅ YAMNet model loaded.")

# ----------- Load Class Labels -----------
print("📁 Loading class labels...")
csv_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
csv_path = 'yamnet_class_map.csv'
if not os.path.exists(csv_path):
    urllib.request.urlretrieve(csv_url, csv_path)
class_names = pd.read_csv(csv_path)['display_name'].tolist()
print("✅ Class labels loaded.")

# ----------- Friendly Name Mapping -----------
friendly_map = {
    "Baby cry, infant cry": "Baby crying",
    "Crying, sobbing": "Crying",
    "Speech": "Talking",
    "Bark": "Dog barking",
    "Meow": "Cat meowing",
    "Laughter": "Laughing",
    "Silence": "Silence",
    "Singing": "Singing",
    "Doorbell": "Doorbell ringing",
    "Fire alarm": "Fire alarm",
    "Knock": "Knocking",
    "Music": "Music",
    "Applause": "Clapping"
}

# ----------- PyAudio Settings -----------
CHUNK = 16000  # 1 second of audio at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

# Choose your input device if needed
DEVICE_INDEX = None  # or set to an integer from p.get_device_info_by_index()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("\n🎧 Listening... Speak or make a sound!")
print("Press Ctrl+C to stop.")

try:
    while True:
        # Read 1 second of audio
        data = stream.read(CHUNK, exception_on_overflow=False)
        waveform = np.frombuffer(data, dtype=np.int16) / 32768.0
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

        # Predict using YAMNet
        scores, embeddings, spectrogram = yamnet_model(waveform_tensor)
        prediction = tf.reduce_mean(scores, axis=0).numpy()

        # Top 5 predictions
        top_indices = prediction.argsort()[-5:][::-1]
        top_labels = [class_names[i] for i in top_indices]
        top_scores = [prediction[i] for i in top_indices]

        # Display top 5
        print("\n📊 Top 5 Predictions:")
        for label, score in zip(top_labels, top_scores):
            print(f"  {label} ({score:.3f})")

        # Friendly label output
        final_label = None
        for label in top_labels:
            if label in friendly_map:
                final_label = friendly_map[label]
                break

        if final_label:
            print(f"🔊 Predicted: {final_label} ✅")
        else:
            print(f"🔊 Predicted: {top_labels[0]} ❓")

except KeyboardInterrupt:
    print("\n🛑 Stopped listening.")
    stream.stop_stream()
    stream.close()
    p.terminate()
