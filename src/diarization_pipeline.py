def main():
    from pathlib import Path
    from dotenv import load_dotenv
    import os
    import certifi
    from diarizer import Diarizer
    from pydub import AudioSegment
    
    os.environ["SSL_CERT_FILE"] = certifi.where()
    load_dotenv()

    audio_path = 'audio/original/Dating Profiles - Matteo Lane & Nick Smith [pwZpBVGfRWY].mp4'

    audio_file = Path(audio_path)

    device = 'cpu'
    batch_size = 4
    compute_type = 'int8'
    model_dir = "../models/"
    # min_speakers = 2
    # max_speakers = 10

    diarizer = Diarizer(token=os.getenv("TOKEN"), device=device, batch_size=batch_size, compute_type=compute_type, model_dir=model_dir)

    # final_result = diarizer.run_pipeline(audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    final_result = diarizer.run_pipeline(audio_file)

    segments = final_result["segments"]

    grouped = []
    current_segment = None

    for segment in segments:
        if current_segment is None:
            current_segment = segment
        elif segment['speaker'] == current_segment['speaker']:
            current_segment['end'] = segment['end']
        else:
            grouped.append(current_segment)
            current_segment = segment

    if current_segment:
        grouped.append(current_segment)

    for segment in grouped:
        print(f"Speaker {segment['speaker']} from {segment['start']} to {segment['end']}")

    audio = AudioSegment.from_file(audio_path)

    for g in grouped:
        start_ms = int(g['start'] * 1000)
        end_ms = int(g['end'] * 1000)
        speaker_id = g['speaker'].replace("SPEAKER_", "SPK")

        clip = audio[start_ms:end_ms]
        filename = f"episode1_{speaker_id}_{start_ms:06d}_{end_ms:06d}.wav"
        path = os.path.join("audio/segments_grouped", filename)
        clip.export(path, format="wav")

if __name__ == "__main__":
    main()