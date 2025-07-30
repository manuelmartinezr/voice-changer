# get the input file from /audio/original
from pathlib import Path
from dotenv import load_dotenv
import whisperx
import gc
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

audio_file = Path('audio/original/Dating Profiles - Matteo Lane & Nick Smith [pwZpBVGfRWY].mp4')

device = 'cpu'
batch_size = 4
compute_type = 'int8'
model_dir = "../models/"
transcribe_model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = transcribe_model.transcribe(audio, batch_size=batch_size)
# print(result['segments'])
gc.collect()
del transcribe_model

align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
print(result['segments'])
gc.collect()
del align_model

diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=os.getenv("TOKEN"))

diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

gc.collect()
del diarize_model

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs