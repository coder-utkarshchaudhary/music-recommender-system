import os
import gc
import cv2 as cv
import numpy as np
import pandas as pd

IMG_SHAPE=64
MFCC_INPUT = r'C:\spectral_features_for_music_recommender_project\mfcc'
SC_INPUT = r'C:\spectral_features_for_music_recommender_project\sc'
SB_INPUT = r'C:\spectral_features_for_music_recommender_project\sb'
ZCR_INPUT = r'C:\spectral_features_for_music_recommender_project\zcr'
OUTPUT = r'spectral_feature_for_training'

def format_name(name_array):
    return [name.split('_')[0] for name in name_array]

def create_X_and_y_for_mfcc(input_path, dataframe, output_path):
    X, y = [], []
    graphs = format_name(os.listdir(input_path))
    
    for idx,graph in enumerate(graphs):
        X.append(cv.resize(cv.imread(os.path.join(input_path), f'{graph}_mfcc_{(idx+1)%20}.png'), (IMG_SHAPE, IMG_SHAPE)))
        y.append([dataframe[f'sentiment_{i+1}', dataframe['name']==graph] for i in range(10)])

    X = np.save(os.path.join(output_path, f'mfcc_X.npy'), np.array(X))
    y = np.save(os.path.join(output_path, f'mfcc_y.npy'), np.array(y))
    print("X and y for mfcc saved.")
    del X
    del y
    gc.collect()

def create_X_and_y_for_sc(input_path, dataframe, output_path):
    X, y = [], []
    graphs = format_name(os.listdir(input_path))

    for graph in graphs:
        X.append(cv.resize(cv.imread(os.path.join(input_path), f'{graph}_spectral_centroid.png'), (IMG_SHAPE, IMG_SHAPE)))
        y.append([dataframe[f'sentiment_{i+1}', dataframe['name']==graph] for i in range(10)])

    X = np.save(os.path.join(output_path, f'sc_X.npy'), np.array(X))
    y = np.save(os.path.join(output_path, f'sc_y.npy'), np.array(y))
    print("X and y for sc saved.")
    del X
    del y
    gc.collect()

def create_X_and_y_for_sb(input_path, dataframe, output_path):
    X, y = [], []
    graphs = format_name(os.listdir(input_path))

    for graph in graphs:
        X.append(cv.resize(cv.imread(os.path.join(input_path), f'{graph}_spectral_band.png'), (IMG_SHAPE, IMG_SHAPE)))
        y.append([dataframe[f'sentiment_{i+1}', dataframe['name']==graph] for i in range(10)])

    X = np.save(os.path.join(output_path, f'sb_X.npy'), np.array(X))
    y = np.save(os.path.join(output_path, f'sb_y.npy'), np.array(y))
    print("X and y for sb saved.")
    del X
    del y
    gc.collect()

def create_X_and_y_for_zcr(input_path, dataframe, output_path):
    X, y = [], []
    graphs = format_name(os.listdir(input_path))

    for graph in graphs:
        X.append(cv.resize(cv.imread(os.path.join(input_path), f'{graph}_zcr.png'), (IMG_SHAPE, IMG_SHAPE)))
        y.append([dataframe[f'sentiment_{i+1}', dataframe['name']==graph] for i in range(10)])

    X = np.save(os.path.join(output_path, f'zcr_X.npy'), np.array(X))
    y = np.save(os.path.join(output_path, f'zcr_y.npy'), np.array(y))
    print("X and y for sc saved.")
    del X
    del y
    gc.collect()

def main(file_name):
    df = pd.read_csv(file_name)
    create_X_and_y_for_mfcc(MFCC_INPUT, df, OUTPUT)
    create_X_and_y_for_sc(SC_INPUT, df, OUTPUT)
    create_X_and_y_for_sb(SB_INPUT, df, OUTPUT)
    create_X_and_y_for_zcr(ZCR_INPUT, df, OUTPUT)

if __name__ =="__main__":
    dataframe_file_path = r''
    main(file_name=dataframe_file_path)