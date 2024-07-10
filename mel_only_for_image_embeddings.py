import os
import librosa
import warnings
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
from image_embeddings import main as embedding_main

IMG_DIM = 20
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")

# Global variables
OUTPUT_PATH = r"C:\spectral_features_for_music_recommender_project\mel"
INPUT_PATH = r"C:\song_files_for_music_recommender_project"
CHECKPOINT_FILE = "processing_checkpoint.txt"

# Helper functions
def process_audio(audio_path):
    audio, sr = librosa.load(audio_path)
    return audio, sr

def plot_mel(mel, sr, output_path, song_name):
    plt.figure(figsize=(IMG_DIM*2, IMG_DIM))
    ax = plt.gca()
    img = librosa.display.specshow(mel, sr=sr, x_axis=None, y_axis=None, ax=ax)
    ax.axis('off')
    plt.savefig(os.path.join(output_path, f"{song_name}_mel.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(index))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    else:
        return 0  # Start from the beginning if checkpoint file doesn't exist

def process_song(song, index):
    try:
        audio_path = os.path.join(INPUT_PATH, f'{song}.mp3')
        audio, sr = process_audio(audio_path)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048)
        mel_db = librosa.power_to_db(mel, ref=np.mean)

        plot_mel(mel_db, sr, OUTPUT_PATH, song)
        print(f"\nMel plotted for {song}...\n")
        save_checkpoint(index)  # Save checkpoint after successful processing

    except Exception as e:
        print(f"\nError processing {song}!!!!!!!!!!\n")

def get_song_title_array(folder_path):
    song_names = os.listdir(folder_path)
    song_names = [song[:-4] for song in song_names if song.endswith('.mp3')]
    return song_names

async def main(song_array):
    start_index = load_checkpoint()
    max_workers = 6  # Adjust this based on your system's capability
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, process_song, song, start_index + idx)
            for idx, song in enumerate(song_array[start_index:], start=start_index)
        ]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    song_array = get_song_title_array(INPUT_PATH)
    asyncio.run(main(song_array))
    embedding_main(img_input=OUTPUT_PATH, json_path=r'image_embeddings.json')

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
