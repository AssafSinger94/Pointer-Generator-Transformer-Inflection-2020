import argparse
import os

parser = argparse.ArgumentParser(description='Reads conll dev file, and covers the target (test file format)')
parser.add_argument('--hall-dir', type=str, default='task0-data/hall',
                    help="Folder of the dataset file")
args = parser.parse_args()


def add_lemma_tag(hall_dir, hall_filename):
    hall_file_path = os.path.join(hall_dir, hall_filename)
    # Get first line
    with open(hall_file_path, "r", encoding='utf-8') as f:
        first_line = f.readline()
    # Split to components
    lemma, target, feature = first_line.replace("\n", "").split("\t")
    # replace to new feature
    pos = feature.split(';')[0]
    feature = f"{pos};LEMMA"
    # Append new line to end of file
    with open(hall_file_path, "a", encoding='utf-8') as f:
        f.write(f"{lemma}\t{lemma}\t{feature}\n")
    pass


if __name__ == '__main__':
    hall_files = os.listdir(args.hall_dir)
    for hall_filename in hall_files:
        add_lemma_tag(args.hall_dir, hall_filename)
