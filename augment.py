import argparse
import os
import shutil
from collections import OrderedDict

import data
import utils

parser = argparse.ArgumentParser(description='Reads conll dataset file, and augments the dataset')
parser.add_argument('--data-dir', type=str, default='train',
                    help="Folder of the dataset file")
parser.add_argument('--aug-dir', type=str, default='aug',
                    help="Folder of the augmented dataset file")
parser.add_argument('--lang', type=str, default='english',
                    help="language of the dataset file")
args = parser.parse_args()

MAX_DATASET_SIZE = 1000000

def group_dataset_by_lemmas(dataset_file_path):
    """ Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""
    train_morph_list = data.read_morph_file(dataset_file_path)
    inflections = OrderedDict()
    for lemma, target, feature in train_morph_list:
        # Get forms for lemma (empty set if not exists yet)
        forms = inflections.get(lemma, set())
        # Add new form
        forms.add((target, feature))
        inflections[lemma] = forms
    for lemma, forms in inflections.items():
        # Get part of speech tag for lemma, same for all inflections of base lemma
        # (by accessing feature in the first tuple and extracting first feature)
        pos = next(iter(forms))[1].split(';')[0]
        forms.add((lemma, f'{pos};LEMMA'))
    return inflections

def augment_dataset(dataset_file_path, aug_file_path):
    # Apply data augmentation to file
    aug_file = open(aug_file_path, "w", encoding='utf-8')  # "ISO-8859-1")
    inflections = group_dataset_by_lemmas(dataset_file_path)
    dataset_size = 0
    # Go over all inflection sets - (word, features)
    for forms in inflections.values():
        for src_pair in forms:
            for tgt_pair in forms:
                dataset_size += 1
                lemma = src_pair[0]
                target, feature = tgt_pair
                aug_file.write(f"{lemma}\t{target}\t{feature}\n")
    aug_file.close()
    print(f"{aug_file_path} - {dataset_size}")
    return dataset_size


def limit_file_size(dataset_file_path, aug_file_path, dataset_size, max_dataset_size):
    reduced_dataset_size = 0
    # Get dataset and augmented file
    dataset_file = open(dataset_file_path, "r", encoding='utf-8')  # "ISO-8859-1")
    # Get reduced file to write to
    reduced_file_path = aug_file_path + ".red"
    reduced_file = open(reduced_file_path, "w", encoding='utf-8')  # "ISO-8859-1")
    # Write whole dataset to reduced file
    lines = dataset_file.readlines()
    for line in lines:
        if line != "\n":
                reduced_file.write(line)
                reduced_dataset_size += 1
    dataset_file.close()
    # Write the rest from augmented file
    keep_line_every = int(round(dataset_size / (max_dataset_size - reduced_dataset_size)))
    aug_file = open(aug_file_path, "r", encoding='utf-8')  # "ISO-8859-1")
    aug_lines = aug_file.readlines()
    line_index = 0
    # round to nearest integer
    # Write line every keep_line_every lines
    for line in aug_lines:
        if line != "\n":
            if line_index % keep_line_every == 0:
                reduced_file.write(line)
                reduced_dataset_size += 1
            line_index += 1
    aug_file.close()
    reduced_file.close()
    # move and overwrite augmented file with reduced file
    shutil.move(reduced_file_path, aug_file_path)
    print(f"{aug_file_path} - {reduced_dataset_size}")

def full_augmentation(data_dir, aug_dir, language):
    # Initialize augmented file path and dir
    utils.maybe_mkdir(aug_dir)
    aug_file_path = os.path.join(aug_dir, f"{language}.aug")
    # Get number of examples train set
    train_file_path = os.path.join(data_dir, f"{language}.trn")
    # Print number of examples in original dataset
    print(f"{train_file_path} - {len(data.read_morph_file(train_file_path))}")
    # Augment dataset and get size of new augmented file
    dataset_size = augment_dataset(train_file_path, aug_file_path)
    # If dataset passes maximum size, throw extra examples
    if dataset_size > MAX_DATASET_SIZE:
        limit_file_size(train_file_path, aug_file_path, dataset_size, MAX_DATASET_SIZE)


if __name__ == '__main__':
    # Create vocab files
    full_augmentation(args.data_dir, args.aug_dir, args.lang)


    # # If less than 1k examples
    # if train_examples_count < 1000:
    #     # add hallucinations upto 10k examples
    #     os.system(f"python hallucinate.py {args.data_dir} {args.lang} --examples 10000 ")
    #     # Use hallucinated file as new train file
    #     train_file_path = os.path.join(data_dir, f"{language}.hall")
    #     # Print number of examples in original dataset
    #     print(f"{train_file_path} - {len(data.read_morph_file(train_file_path))}")