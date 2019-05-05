from dreaming.streaming import *
from dreaming.audio_output import AudioLoop
import numpy as np
import pyaudio


p = pyaudio.PyAudio()

signal = np.zeros(64000).astype(np.int16)

audio_loop = AudioLoop(signal)


stream = p.open(format=p.get_format_from_width(2, unsigned=False),
                channels=1,
                rate=16000,
                output=True,
                stream_callback=audio_loop.callback,
                frames_per_buffer=4096)

stream.start_stream()

client = LoopStreamClient(port=2222, message_length=128000)
client.start_client()

while True:
    new_data = client.get_new_data()
    if new_data is not None:
        print("new data received")
        new_signal = np.frombuffer(new_data, dtype=np.int16)
        audio_loop.set_new_signal(new_signal)