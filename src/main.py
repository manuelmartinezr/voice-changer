def main():
    from pathlib import Path
    from dotenv import load_dotenv
    import os
    import certifi
    from diarizer import Diarizer
    
    os.environ["SSL_CERT_FILE"] = certifi.where()
    load_dotenv()

    audio_file = Path('audio/original/Dating Profiles - Matteo Lane & Nick Smith [pwZpBVGfRWY].mp4')

    device = 'cpu'
    batch_size = 4
    compute_type = 'int8'
    model_dir = "../models/"

    diarizer = Diarizer(token=os.getenv("TOKEN"), device=device, batch_size=batch_size, compute_type=compute_type, model_dir=model_dir)

    final_result = diarizer.run_pipeline(audio_file)

    print(final_result["segments"])

if __name__ == "__main__":
    main()