import librosa
import os
import gc
import warnings
import numpy as np
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

IMG_DIM = 30
warnings.filterwarnings("ignore")

# Helper functions
def process_audio(audio_path):
    audio, sr = librosa.load(audio_path)
    return audio, sr

def generate_mfcc_graphs(mfcc, output_path, song_name):
    for i in range(20):
        _, ax = plt.subplots(figsize=(IMG_DIM*2, IMG_DIM))
        ax.plot(mfcc[i])
        ax.axis('off')
        plt.savefig(os.path.join(output_path, f"{song_name}_mfcc_{i+1}.png"))
        plt.close()

def generate_spectral_centroid_graphs(spectral_centroid, output_path, song_name):
    _, ax = plt.subplots(figsize=(IMG_DIM*2, IMG_DIM))
    ax.plot(spectral_centroid[0])
    ax.axis('off')
    plt.savefig(os.path.join(output_path, f"{song_name}_spectral_centroid.png"))
    plt.close()

def generate_spectral_bandwidth_graphs(spectral_band, output_path, song_name):
    _, ax = plt.subplots(figsize=(IMG_DIM*2, IMG_DIM))
    ax.plot(spectral_band[0])
    ax.axis('off')
    plt.savefig(os.path.join(output_path, f"{song_name}_spectral_band.png"))
    plt.close()

def generate_zcr_graphs(zcr, output_path, song_name):
    _, ax = plt.subplots(figsize=(IMG_DIM*2, IMG_DIM))
    ax.plot(zcr[0])
    ax.axis('off')
    plt.savefig(os.path.join(output_path, f"{song_name}_zcr.png"))
    plt.close()

def get_song_title_array(folder_path):
    song_names = os.listdir(folder_path)
    for idx in range(len(song_names)):
        if song_names[idx][-4:] == '.mp3':
            song_names[idx] = song_names[idx][:-4]
    return song_names

OUTPUT_PATH = r"C:\spectral_features_for_music_recommender_project"
INPUT_PATH = r"C:\song_files_for_music_recommender_project"

def process_song(song):
    try:
        audio_path = os.path.join(INPUT_PATH, f'{song}.mp3')
        audio, sr = process_audio(audio_path)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=128)
        mel_db = librosa.power_to_db(mel, ref=np.mean)

        mfcc = librosa.feature.mfcc(S=mel_db, sr=sr)
        sc = librosa.feature.spectral_centroid(S=mel, sr=sr)
        sb = librosa.feature.spectral_bandwidth(S=mel, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)

        generate_mfcc_graphs(mfcc, os.path.join(OUTPUT_PATH, 'mfcc'), song)
        print(f"MFCC for {song} stored")
        del mfcc

        generate_spectral_centroid_graphs(sc, os.path.join(OUTPUT_PATH, 'sc'), song)
        print(f"SC for {song} stored")
        del sc

        generate_spectral_bandwidth_graphs(sb, os.path.join(OUTPUT_PATH, 'sb'), song)
        print(f"SB for {song} stored")
        del sb

        generate_zcr_graphs(zcr, os.path.join(OUTPUT_PATH, 'zcr'), song)
        print(f"ZCR for {song} stored")
        del zcr

        del mel
        del mel_db
        gc.collect()

    except Exception as e:
        print(f"Error processing {song}: {e}")

def main(song_array):
    max_workers = 10  # Adjust this based on your system's capability
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_song, song): song for song in song_array}
        for future in as_completed(futures):
            song = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {song}: {e}")

if __name__ == '__main__':
    main(get_song_title_array(INPUT_PATH))
