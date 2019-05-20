from dreaming.streaming import *
import torch
import time
import numpy as np
import pickle

# server = LoopStreamServer(port=2222, message_length=64000)
# server.start_server()
#
# time.sleep(0.5)
#
# client = LoopStreamClient(port=2222, message_length=64000)
# client.start_client()
#
# test_data = np.random.randn(16000).astype(np.float32)
# test_bytes = test_data.tobytes()
#
# server.set_data(test_bytes)
#
# #time.sleep(0.5)
# while client.get_new_data() is None:
# print("not yet received")
# time.sleep(0.01)
#
# server.stop()
# client.stop()
#
# print("read received bytes")
# received_bytes = client.finished_data
# print("received data:", np.frombuffer(received_bytes, dtype=np.float32))

test_dict = {"data": np.random.rand(1000, 1000),
             "sub_dict": {"data_1": np.random.randint(0, 10, 100),
                          "data_2": np.random.randint(0, 1, 10)}}

tic = time.time()
pickled_dict = pickle.dumps(test_dict)
toc = time.time()

print("duration to pickle a dict with", len(pickled_dict), "bytes:", toc-tic)

server = SocketDataExchangeServer(port=2222)
time.sleep(0.5)

client = SocketDataExchangeClient(port=2222)
time.sleep(0.5)

tic = time.time()
server.set_new_data(pickled_dict)

received_message = None
while received_message is None:
    time.sleep(0.01)
    received_message = client.get_received_data()
    toc = time.time()

print("transmission duration:", toc-tic)

tic = time.time()
unpickled_dict = pickle.loads(received_message)
toc = time.time()

print("duration to unpickle:", toc-tic)

print(unpickled_dict)

server.stop()
client.stop()


