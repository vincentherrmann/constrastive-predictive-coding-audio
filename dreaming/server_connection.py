import socket
import pyaudio
import numpy as np
import io
import time

# Socket
HOST = '127.0.0.1' #socket.gethostname()
PORT = 2222 #8765

# Audio
#p = pyaudio.PyAudio()
CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
#stream = p.open(format=FORMAT,
#                channels=CHANNELS,
#                rate=RATE,
#                output=True,
#                frames_per_buffer=CHUNK)


test_audio = np.random.randn(1, 16000).astype(np.float32)
print(test_audio)
test_bytes = io.BytesIO(test_audio.tobytes())
print("num test_bytes:", len(test_bytes.getvalue()))
received_bytes = io.BytesIO()

#data = test_bytes.read(4096)
#while data is not b'':
#    received_bytes.write(data)
#    data = test_bytes.read(4096)
    #print(data)

#print("num received_bytes:", len(received_bytes.getvalue()))
#received_audio = np.frombuffer(received_bytes.getvalue(), dtype=np.float32)

with socket.socket() as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    conn, address = server_socket.accept()
    print("Connection from " + address[0] + ":" + str(address[1]))

    data = test_bytes.read(4096)
    while data is not b'':
        conn.send(data)
        data = test_bytes.read(4096)
    conn.send(b'')

    time.sleep(5)

    #data = conn.recv(4096)
    #while data != "":
    #    data = conn.recv(4096)
    #    print(data)

#stream.stop_stream()
#stream.close()
#p.terminate()