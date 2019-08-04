import torchaudio
import mutagen.mp3

file = mutagen.mp3.MP3("/Users/vincentherrmann/Desktop/test_file.mp3")
print("length:", file.info.length * file.info.sample_rate)

data, _ = torchaudio.load("/Users/vincentherrmann/Desktop/test_file.mp3")
pass