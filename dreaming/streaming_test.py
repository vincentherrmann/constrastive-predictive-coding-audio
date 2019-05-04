from dreaming.streaming import *
import torch
import time
import numpy as np

server = LoopStreamServer(port=2222, message_length=64000)
server.start_server()

time.sleep(0.5)

client = LoopStreamClient(port=2222, message_length=64000)
client.start_client()

test_data = np.random.randn(16000).astype(np.float32)
test_bytes = test_data.tobytes()

server.set_data(test_bytes)

#time.sleep(0.5)
while client.get_new_data() is None:
    print("not yet received")
    time.sleep(0.01)

server.stop()
client.stop()

print("read received bytes")
received_bytes = client.finished_data
print("received data:", np.frombuffer(received_bytes, dtype=np.float32))

