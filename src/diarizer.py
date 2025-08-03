import whisperx
class Diarizer:
    def __init__(self, token, device='cpu', batch_size=4, compute_type='int8', model_dir="../models/"):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.audio = None
        self.token = token

    def load_audio(self, audio_file):
        self.audio = whisperx.load_audio(audio_file)

    def transcribe(self):
        transcribe_model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, download_root=self.model_dir)
        result = transcribe_model.transcribe(self.audio, batch_size=self.batch_size)
        return result
    
    # align words more precisely with the audio
    def align(self, transcription):
        align_model, metadata = whisperx.load_align_model(language_code=transcription["language"], device=self.device)
        aligned_result = whisperx.align(transcription["segments"], align_model, metadata, self.audio, self.device, return_char_alignments=False)
        return aligned_result
    
    # get speaker time stamps
    def diarize(self, min_speakers=None, max_speakers=None):
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=self.token)
        diarized_segments = diarize_model(self.audio, min_speakers=min_speakers, max_speakers=max_speakers)
        return diarized_segments
        
    def run_pipeline(self, audio_file, min_speakers=None, max_speakers=None):
        self.load_audio(audio_file)
        transcription = self.transcribe()
        aligned_result = self.align(transcription)
        diarized_segments = self.diarize(min_speakers, max_speakers)
        final_result = whisperx.assign_word_speakers(diarized_segments, aligned_result) # asign speakers to words
        return final_result
# segments are now assigned speaker IDs

        