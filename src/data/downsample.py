import os
import random

MAX_DATASET_SIZE = 100
DATA_DIR = "task0-data/out"
DOWNSAMPLE_DIR = "task0-data/small"
DOWNSAMPLE_LANGUAGES = ['mlg', 'ceb', 'hil', 'mao', 'gmh', 'kon', 'lin', 'sot', 'zul', 'gaa',
 'zpv', 'izh', 'mwf', 'tel', 'gml', 'tgk', 'dje', 'xno', 'kjh', 'lud', 'vro']

def downsample_dataset(lang):
    # Get file path for train and small dataset
    train_file_path = os.path.join(DATA_DIR, f"{lang}.trn")
    small_file_path = os.path.join(DOWNSAMPLE_DIR, f"{lang}.sml")
    # Get examples of train set and size
    train_file = open(train_file_path, "r", encoding='utf-8')
    train_lines = train_file.readlines()
    train_set_size = len(train_lines)
    try:
        lines_to_keep = random.sample(range(0, train_set_size), MAX_DATASET_SIZE)
    except ValueError:
        lines_to_keep = range(0, train_set_size)
    # Monitor progress
    print(f"downsampling file {train_file_path}: original size {train_set_size}, downsample size: {len(set(lines_to_keep))}")
    with open(small_file_path, "w", encoding='utf-8') as fp:
        # For each example in prediction file
        for i in sorted(lines_to_keep):
            print(train_lines[i].strip("\n"), file=fp)


if __name__ == '__main__':
    if not os.path.exists(DOWNSAMPLE_DIR):
        os.makedirs(DOWNSAMPLE_DIR)
    for lang in sorted(DOWNSAMPLE_LANGUAGES):
        downsample_dataset(lang)