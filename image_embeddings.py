from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embeddings(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    return outputs.cpu().numpy().tolist()

def get_song_name(song):
    name = song.split('_')[0]
    return name

def main(img_input, json_path):
    embeddings_data = []

    images = os.listdir(img_input)
    i = 1
    for image in images:
        try:
            if image.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(img_input, image)
                embeddings = get_image_embeddings(image_path)
                song_name = get_song_name(image)
                data = {
                    "name": song_name,
                    "embedding": embeddings
                }
                embeddings_data.append(data)
                print(f"{i}. Embeddings done for song : {song_name}")
                # print(f"{len(embeddings_data)}\n")
                i+=1
        except:
            print(f"{i}. Problem processing song : {song_name}")
            i+=1

    with open(json_path, 'w') as file:
        json.dump(embeddings_data, file, indent=4)

    print("\nData saved to json.")

if __name__ == '__main__':
    img_input = r"C:\spectral_features_for_music_recommender_project\mel"
    json_path = r"image_embeddings.json"
    main(img_input, json_path)