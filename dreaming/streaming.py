import socket
import time
#import pyaudio
import numpy as np
import io
import threading


class LoopStreamServer:
    def __init__(self, port, host='127.0.0.1', message_length=4096):
        self.port = port
        self.host = host
        self.streaming_thread = threading.Thread()
        self.lock = threading.Lock()
        self.new_data = None
        self.current_data = None
        self.stream = False
        self.message_length = message_length
        self.chunk_size = 4096

    def start_server(self):
        self.stream = True
        self.streaming_thread = threading.Thread(target=self.serve)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

    def serve(self):
        with socket.socket() as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server_socket.bind((self.host, self.port))
            except:
                print("server can't connect")
                return
            server_socket.listen(1)

            conn, address = server_socket.accept()
            print("Connection from " + address[0] + ":" + str(address[1]))
            while self.stream:
                if self.current_data is not None:
                    current_bytes = self.current_data.read(self.chunk_size)
                    if len(current_bytes) == 0:
                        self.current_data = None
                        continue
                else:
                    with self.lock:
                        self.current_data = self.new_data
                        self.new_data = None
                    time.sleep(0.1)
                    continue
                conn.send(current_bytes)
                #print("send", len(current_bytes), "bytes")

    def stop(self):
        self.stream = False
        self.streaming_thread.join()

    def set_data(self, data):
        with self.lock:
            assert len(data) == self.message_length
            self.new_data = io.BytesIO(data)


class LoopStreamClient:
    def __init__(self, port, host='127.0.0.1', message_length=4096):
        self.port = port
        self.host = host
        self.streaming_thread = threading.Thread()
        self.lock = threading.Lock()
        self.current_data = io.BytesIO()
        self.finished_data = None
        self.stream = False
        self.new_data_available = False
        self.message_length = message_length
        self.position_in_message = 0
        self.chunk_size = 4096

    def start_client(self):
        self.stream = True
        self.streaming_thread = threading.Thread(target=self.receive)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

    def receive(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                client_socket.connect((self.host, self.port))
            except:
                print("client can't connect")
                return
            while self.stream:
                remaining_message_length = self.message_length - self.position_in_message
                last_chunk = remaining_message_length <= self.chunk_size
                receive_size = remaining_message_length if last_chunk else self.chunk_size
                new_bytes = client_socket.recv(receive_size)
                if new_bytes == b'':
                    if self.position_in_message == 0:
                        continue
                    else:
                        print("connection from server lost")
                        return
                    #continue
                #print("receive", len(new_bytes), "bytes")
                if last_chunk:
                    if self.current_data.getvalue() != b'':
                        self.current_data.write(new_bytes)
                        with self.lock:
                            self.finished_data = self.current_data.getvalue()
                            self.new_data_available = True
                        self.current_data = io.BytesIO()
                    self.position_in_message = 0
                    continue
                self.current_data.write(new_bytes)
                self.position_in_message += receive_size

    def get_new_data(self):
        with self.lock:
            if self.new_data_available:
                self.new_data_available = False
                return self.finished_data
        return None

    def stop(self):
        self.stream = False
        self.streaming_thread.join()
