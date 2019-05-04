import socket
import io
import numpy as np
#import pyaudio

# Socket
HOST = '127.0.0.1' #socket.gethostname()
PORT = 2222

# Audio
CHUNK = 1024 * 4
#FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
#p = pyaudio.PyAudio()
#stream = p.open(format=FORMAT,
#                channels=CHANNELS,
#                rate=RATE,
#                input=True,
#                frames_per_buffer=CHUNK)

print("Recording")

received_bytes = io.BytesIO()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    print("connected")

    data = 'abcdefg'.encode()
    client_socket.send(data)

    data = client_socket.recv(4096)
    while data != b'':
        received_bytes.write(data)
        data = client_socket.recv(4096)

    received_audio = np.frombuffer(received_bytes.getvalue(), dtype=np.float32)
    print("received_audio:", received_audio)