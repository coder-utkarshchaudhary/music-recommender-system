import pandas as pd
import os
import warnings
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from feature_extraction import ZeroShotClassifier

warnings.filterwarnings("ignore")
classifier = ZeroShotClassifier()

sentiments = ['amusement', 'anxiety', 'angry', 'dreaminess', 'eroticism', 'feeling pumped up', 'joy', 'relaxation', 'romance', 'sadness']

def get_song_title_array(folder_path):
    song_names = os.listdir(folder_path)
    for idx in range(len(song_names)):
        if song_names[idx][-4:] == '.mp3':
            song_names[idx] = song_names[idx][:-4]
    return song_names

def get_lyrics(data, song_title):
    lyrics = None
    for item in data:
        if item['title'] == song_title:
            lyrics = item.get('lyrics', None)
            break
    return lyrics

def get_sentiment(text):
    sentiment = classifier.classify(text, sentiments, multi_label=True, return_logits=True)
    return sentiment

def store_lyrics(dataframe, song_name, lyrics, sentiments):
    new_row = {'name': song_name, 'lyrics': lyrics}
    for i, sentiment in enumerate(sentiments):
        new_row[f'sentiment_{i+1}'] = sentiment
    dataframe = dataframe._append(new_row, ignore_index=True)
    return dataframe

def process_song(track_details, song_data):
    lyrics = get_lyrics(song_data, track_details)
    if lyrics:
        sentiments = get_sentiment(lyrics)
        return track_details, lyrics, sentiments
    else:
        return track_details, None, None

def main(song_data, output_path):
    columns = ['name', 'lyrics'] + [f'sentiment_{i+1}' for i in range(10)]
    df = pd.DataFrame(columns=columns)
    
    song_names = get_song_title_array(r'C:\song_files_for_music_recommender_project')

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_song, track_details, song_data) for track_details in song_names]
        
        for future in as_completed(futures):
            track_details, lyrics, sentiments = future.result()
            if lyrics:
                df = store_lyrics(df, track_details, lyrics, sentiments)
                print(f"DataFrame updated for {track_details}")
                print(f"\nShape of dataframe rn is : {df.shape}\n")
            else:
                print(f"Lyrics not found for {track_details}")

    # Storing the DataFrame
    df.to_csv(os.path.join(output_path, 'lyrics_vs_sentiment.csv'), index=False)
    print("\n\nDataFrame stored. Check")

if __name__ == "__main__":
    with open(r'song_data.json', 'r') as song_file:
        song_data = json.load(song_file)

    main(song_data=song_data, output_path=r'lyrics_arrays/')