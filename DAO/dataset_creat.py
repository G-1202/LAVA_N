import numpy as np
import pandas as pd
import os
import subprocess
import shutil
from ultralytics import YOLO
from collections import defaultdict
from DAO_utils import process_all_folders

VIDEO_BIT_RATE = [200, 400, 800, 1200, 2200, 3300, 5000, 6500, 8600, 10000, 12000]
resolutions = ['480p', '720p', '1080p']
button = 33
re = [[720, 480], [1280, 720], [1920, 1080]]
Fps = 30
segment_length = 1
frames_per_segment = Fps * segment_length
raw_data = 'raw_data'
mid_folder = 'mid'
dataset = 'dataset/train'


if __name__ == '__main__':
    # Create training data
    f1_scores = process_all_folders(raw_data, mid_folder, dataset, frames_per_segment, Fps)
    # Constructing the column names for the CSV
    column_names = ['frame_path'] + [f'f1_score_{bitrate}_{res}' for res in resolutions for bitrate in VIDEO_BIT_RATE]

    # Creating the frame_path column
    frame_paths = [f"{i:04}.JPEG" for i in range(1, len(f1_scores)+1)]  # Assuming file names are 001.png, 002.png, ...

    # Combining frame_paths with f1_scores into a single DataFrame
    data = np.column_stack((frame_paths, f1_scores))
    df = pd.DataFrame(data, columns=column_names)

    # Save the DataFrame to a CSV file
    csv_file_path = 'dataset/train/scores.csv'
    df.to_csv(csv_file_path, index=False)
