import pyaudio
import time
import sys
import numpy as np
import threading


#signal = np.sin(np.linspace(0, np.pi*1000, 16000, endpoint=False)) * 30000
#signal = signal.astype(np.int16)

#signal_new = np.sin(np.linspace(0, np.pi*1500, 16000, endpoint=False)) * 30000
#signal_new = signal_new.astype(np.int16)

#p = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, signal, crossfade=2000):
        self.signal = signal
        self.signal_length = signal.shape[0]
        self.new_signal = signal
        self.position = 0
        self.lock = threading.Lock()
        self.new_signal_available = False
        self.crossfade = np.linspace(0, 1, crossfade)
        self.crossfade_length = crossfade

    def callback(self, in_data, frame_count, time_info, status):
        #print(self.position)
        self.position += frame_count
        with self.lock:
            if self.position < self.signal_length - self.crossfade_length:
                data = self.signal[self.position-frame_count:self.position].tobytes()
            else:
                if not self.new_signal_available:
                    data_1 = self.signal[self.position-frame_count:].tobytes()
                    self.position = self.position % self.signal_length
                    data_2 = self.signal[:self.position].tobytes()
                    data = b''.join([data_1, data_2])
                else:
                    start_position = self.position - frame_count
                    crossfade_start = self.signal_length - self.crossfade_length

                    chunk_length = 0

                    # before crossfade
                    data_before_crossfade = b''
                    if start_position < crossfade_start:
                        data_before_crossfade = self.signal[start_position:crossfade_start]
                        chunk_length += crossfade_start - start_position

                    # crossfade
                    crossfade_offset = max(0, start_position - crossfade_start)
                    crossfade_length = min(self.crossfade_length, frame_count - chunk_length)
                    crossfade = self.crossfade[crossfade_offset:crossfade_offset+crossfade_length]
                    crossfade_data_old = self.signal[crossfade_start+crossfade_offset:crossfade_start+crossfade_offset+crossfade_length]
                    crossfade_data_new = self.new_signal[crossfade_start + crossfade_offset:crossfade_start + crossfade_offset + crossfade_length]
                    crossfade_data = (crossfade_data_old * (1. - crossfade)) + (crossfade_data_new * crossfade)
                    crossfade_data = crossfade_data.astype(np.int16)

                    # after crossfade
                    data_after_crossfade = b''
                    if self.position >= self.signal_length:
                        self.position = self.position % self.signal_length
                        self.signal = self.new_signal
                        self.new_signal_available = False
                        print("use new signal")
                        data_after_crossfade = self.signal[:self.position]

                    data = b''.join([data_before_crossfade, crossfade_data, data_after_crossfade])
        return data, pyaudio.paContinue

    def set_new_signal(self, new_signal):
        if new_signal.shape[0] != self.signal_length:
            print("error: new signal has different length")
            return
        with self.lock:
            self.new_signal_available = True
            self.new_signal = new_signal


#audio_loop = AudioLoop(signal)


# stream = p.open(format=p.get_format_from_width(2, unsigned=False),
#                 channels=1,
#                 rate=16000,
#                 output=True,
#                 stream_callback=audio_loop.callback,
#                 frames_per_buffer=4096)
#
# stream.start_stream()
#
# time.sleep(1.)
#
# audio_loop.set_new_signal(signal_new)
#
# time.sleep(2.)
#
# audio_loop.set_new_signal(signal)
# audio_loop.set_new_signal(signal_new)
# audio_loop.set_new_signal(signal)
#
# time.sleep(2.)
#
# audio_loop.set_new_signal(signal_new)
#
# time.sleep(2.)
#
# audio_loop.set_new_signal(signal)
#
# time.sleep(2.)
#
# stream.stop_stream()
# stream.close()
#
# p.terminate()