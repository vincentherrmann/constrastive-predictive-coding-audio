from unittest import TestCase
from dreaming.dreaming_functions import *
from dreaming.streaming import *
import torch
import time
import numpy as np


class TestJitter(TestCase):
    def test_jitter(self):
        x = torch.rand(1, 5, 7)
        jitter_module = Jitter(jitter_sizes=[2, 3], dims=[1, 2], jitter_batches=3)
        jitter_module.verbose = 1
        jx = jitter_module(x)
        print(x)
        print(jx)

    def test_jitter_indexing(self):
        x = torch.rand(1, 5, 7)


class TestStreaming(TestCase):
    def test_loop_streaming(self):
        server = LoopStreamServer(port=2222)
        server.start_server()

        time.sleep(1.)

        client = LoopStreamClient(port=2222)
        client.start_client()

        test_data = np.random.randn(16000).astype(np.float32)
        test_bytes = test_data.tobytes()

        server.set_data(test_bytes)

        time.sleep(0.5)
        #while client.get_new_data() is None:
        #    print("not yet received")
        #    time.sleep(0.1)

        print("read received bytes")
        received_bytes = client.get_new_data()
        print("received data:", np.frombuffer(received_bytes, dtype=np.float32))

        server.stop()
        client.stop()

