import os
import random
import shutil
from collections import OrderedDict

TASK0_LANGS = ['aka', 'ang', 'ast', 'aze', 'azg', 'bak', 'ben', 'bod', 'cat', 'ceb', 'cly', 'cpa', 'cre', 'crh', 'ctp',
               'czn', 'dak', 'dan', 'deu', 'dje', 'eng', 'est', 'evn', 'fas', 'fin', 'frm', 'frr', 'fur', 'gaa', 'glg',
               'gmh', 'gml', 'gsw', 'hil', 'hin', 'isl', 'izh', 'kan', 'kaz', 'kir', 'kjh', 'kon', 'kpv', 'krl', 'lin',
               'liv', 'lld', 'lud', 'lug', 'mao', 'mdf', 'mhr', 'mlg', 'mlt', 'mwf', 'myv', 'nld', 'nno', 'nob', 'nya',
               'olo', 'ood', 'orm', 'ote', 'otm', 'pei', 'pus', 'san', 'sme', 'sna', 'sot', 'swa', 'swe', 'syc', 'tel',
               'tgk', 'tgl', 'tuk', 'udm', 'uig', 'urd', 'uzb', 'vec', 'vep', 'vot', 'vro', 'xno', 'xty', 'zpv', 'zul']

MAX_DATASET_SIZE = 1000000
DATA_DIR = 'task0-data/out'
AUG_DIR = 'task0-data/aug'

def read_morph_file(morph_file_path):
    """ Reads conll file, split to line, and splits each line by tabs. Returns list of lists"""
    # Get all lines in file
    morph_file = open(morph_file_path, 'r', encoding='utf-8')
    lines = morph_file.readlines()
    outputs = []
    # Separate lines to proper format
    for line in lines:
        if line != "\n":
            # Strip '\n' and split
            outputs.append(line.replace("\n", "").split("\t"))
    morph_file.close()
    return outputs

def group_dataset_by_lemmas(dataset_file_path):
    """ Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""
    train_morph_list = read_morph_file(dataset_file_path)
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

def multitask_augment(dataset_file_path, aug_file_path):
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

def downsample_dataset(train_file_path, aug_file_path, max_dataset_size):
    reduced_dataset_size = 0
    # Get dataset and augmented file
    train_file = open(train_file_path, "r", encoding='utf-8')
    # Get reduced file to write to
    reduced_file_path = aug_file_path + ".red"
    reduced_file = open(reduced_file_path, "w", encoding='utf-8')
    # Write whole dataset to reduced file
    lines = train_file.readlines()
    for line in lines:
        if line != "\n":
            reduced_file.write(line)
            reduced_dataset_size += 1

    # Get number of lines to keep from augmented set
    count_lines_to_keep = max_dataset_size - reduced_dataset_size
    # Get examples of augmented set and size
    aug_file = open(aug_file_path, "r", encoding='utf-8')
    aug_lines = aug_file.readlines()
    aug_set_size = len(aug_lines)
    try:
        lines_to_keep = random.sample(range(0, aug_set_size), count_lines_to_keep)
    except ValueError:
        lines_to_keep = range(0, aug_set_size)
    # Monitor progress
    print(f"downsampling file {aug_file_path}: original size {aug_set_size}, downsample size: {len(set(lines_to_keep))}")
    # For each example in prediction file
    for i in sorted(lines_to_keep):
        reduced_file.write(aug_lines[i])
    # close files
    train_file.close()
    reduced_file.close()
    aug_file.close()
    # move and overwrite augmented file with reduced file
    shutil.move(reduced_file_path, aug_file_path)
    print(f"Dataset: {aug_file_path}, Size: {reduced_dataset_size}")




def augment_dataset():
    for lang in TASK0_LANGS:
        # Initialize raw and augmented file paths
        aug_file_path = os.path.join(AUG_DIR, f"{lang}.aug")
        train_file_path = os.path.join(DATA_DIR, f"{lang}.trn")
        # Print number of examples in original dataset
        print(f"{train_file_path} - {len(read_morph_file(train_file_path))}")
        # Augment dataset and get size of new augmented file
        dataset_size = multitask_augment(train_file_path, aug_file_path)
        # If dataset passes maximum size, throw extra examples
        if dataset_size > MAX_DATASET_SIZE:
            downsample_dataset(train_file_path, aug_file_path, MAX_DATASET_SIZE)


if __name__ == '__main__':
    if not os.path.exists(AUG_DIR):
        os.makedirs(AUG_DIR)
    # Create vocab files
    augment_dataset()
