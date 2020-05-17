import argparse
import os
import random

parser = argparse.ArgumentParser(description='Reads conll dataset file, and augments the dataset')
parser.add_argument('--data-dir', type=str, default='train',
                    help="Folder of the dataset file")
args = parser.parse_args()

DOWNSAMPLE_LANGUAGES = ['mlg', 'ceb', 'hil', 'mao', 'gmh', 'kon', 'lin', 'sot', 'zul', 'gaa',
 'zpv', 'izh', 'mwf', 'tel', 'gml', 'tgk', 'dje', 'xno', 'kjh', 'lud', 'vro']
MAX_DATASET_SIZE = 100


def downsample_dataset(data_dir, downsample_dir, lang):
    # Get file path for train and small dataset
    train_file_path = os.path.join(data_dir, f"{lang}.trn")
    small_file_path = os.path.join(downsample_dir, f"{lang}.sml")
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
    langs = sorted(DOWNSAMPLE_LANGUAGES)
    downsample_dir = "task0-data/small"
    if not os.path.exists(downsample_dir):
        os.makedirs(downsample_dir)
    for lang in langs:

        downsample_dataset(args.data_dir, downsample_dir, lang)