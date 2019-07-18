import socket
import socketserver
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
        self.chunk_size = 16384

    def start_server(self):
        self.stream = True
        self.streaming_thread = threading.Thread(name='loop stream', target=self.serve)
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
        self.chunk_size = 16384

    def start_client(self):
        self.stream = True
        self.streaming_thread = threading.Thread(name='loop stream', target=self.receive)
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


class SocketDataExchange:
    def __init__(self):
        self.sending_thread = threading.Thread()
        self.receiving_thread = threading.Thread()
        self.receive_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.received_data = None
        self.sending_data = None
        self.stream = False
        self.new_data_available = False
        self.sending_data_available = False
        self.connection = None
        self.chunk_size = 16384

    def receive(self):
        while self.stream:
            message_length = int.from_bytes(self.connection.recv(64), byteorder='big', signed=False)
            target_data_length = message_length
            if message_length == 0:
                continue
                time.sleep(0.01)
            print("receiving message with size", message_length)
            tik = time.time()
            data_io = io.BytesIO()
            data_writer = io.BufferedWriter(data_io, message_length)
            chunk_number = 0
            while message_length > 0:
                receive_size = min(self.chunk_size, message_length)
                chunk = self.connection.recv(receive_size)
                data_writer.write(chunk)
                receive_size = len(chunk)
                message_length -= receive_size
                #time.sleep(0.01)

                # if len(chunk) != self.chunk_size:
                #     print("chunk", chunk_number, "has size", len(chunk))

                chunk_number += 1

            #print("num chunks:", chunk_number, "last chunk size:", receive_size)
            #print("left message length:", message_length)

            data_writer.flush()
            data = data_io.getvalue()
            data_io.close()

            #data_bytes = data.getvalue()
            if len(data) == target_data_length:
                print("message received")
                duration = time.time() - tik
                if duration > 1.:
                    print("it took", duration, "seconds")
            else:
                print("wrong received message length:", len(data))
                #raise

            with self.receive_lock:
                self.received_data = data
                self.new_data_available = True

    def get_received_data(self):
        with self.receive_lock:
            if self.new_data_available:
                self.new_data_available = False
                return self.received_data
        return None

    def send(self):
        while self.sending_data_available:
            with self.send_lock:
                self.sending_data_available = False
                data = self.sending_data
            message_length = len(data).to_bytes(64, byteorder='big', signed=False)
            if self.connection is None:
                print("No connection")
                return
            self.connection.send(message_length)
            print("sending message with size", len(data))
            self.connection.sendall(data)

    def set_new_data(self, data):
        with self.send_lock:
            self.sending_data_available = True
            self.sending_data = data
        if not self.sending_thread.isAlive():
            self.sending_thread = threading.Thread(name='stream receive', target=self.send)
            self.sending_thread.daemon = True
            self.sending_thread.start()

    def start(self):
        if self.connection is None:
            print("can't start streaming without connection")
            return
        self.stream = True
        self.receiving_thread = threading.Thread(name='stream send', target=self.receive)
        self.receiving_thread.daemon = True
        self.receiving_thread.start()
        print("streaming is active")

    def stop(self):
        self.stream = False
        try:
            self.receiving_thread.join(timeout=0.5)
        except:
            pass

        try:
            self.sending_thread.join(timeout=0.5)
        except:
            pass


class SocketDataExchangeServer(SocketDataExchange):
    def __init__(self, port, host='127.0.0.1', stream_automatically=True):
        super().__init__()
        self.port = port
        self.host = host
        self.socket = socket.socket()
        self.stream_automatically = stream_automatically

        connect_thread = threading.Thread(name='connect ot client', target=self.connect_to_client)
        connect_thread.daemon = True
        connect_thread.start()

    def connect_to_client(self):
        try:
            self.socket.bind((self.host, self.port))
            print("server created")
        except:
            print("server cannot bind")
            return
        self.socket.listen(1)

        conn, address = self.socket.accept()
        print("Connection from " + address[0] + ":" + str(address[1]))
        self.connection = conn

        if self.stream_automatically:
            self.start()

    def stop(self):
        super().stop()
        self.socket.close()


class SocketDataExchangeClient(SocketDataExchange):
    def __init__(self, port, host='127.0.0.1', stream_automatically=True):
        super().__init__()
        self.port = port
        self.host = host
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.stream_automatically = stream_automatically

        try:
            self.socket.connect((self.host, self.port))
            self.connection = self.socket
            print("client has successfully connected")
        except:
            print("client can't connect")
            return

        if self.stream_automatically:
            self.start()

    def stop(self):
        super().stop()
        self.socket.close()

