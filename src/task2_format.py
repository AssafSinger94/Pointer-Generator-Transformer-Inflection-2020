import argparse
import collections
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Gets list of prediction files, and outputs majority voting ensemble '
                                             'pred file')
parser.add_argument('--data-dir', type=str, help="dir of models checkpoints")
parser.add_argument('--pred-dir', type=str, help="dir of models checkpoints")

args = parser.parse_args()
TASK2_LANGUAGES = ['Basque', 'Bulgarian', 'English', 'Finnish', 'German', 'Kannada', 'Maltese',
                   'Navajo', 'Persian', 'Portuguese', 'Russian', 'Spanish', 'Swedish', 'Turkish']

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

def read_test_file(pred_file_path):
    """ Reads conll file, Returns only targets"""
    # Get all lines in file
    base_pred_file = open(pred_file_path, 'r', encoding='utf-8')
    lines = base_pred_file.readlines()
    preds = []
    for line in lines:
        if not line or line == "\n" or line.startswith("prediction"):
            continue
        # Strip '\n' and split, add only first component (predictions)
        preds.append(line.replace("\n", ""))
    base_pred_file.close()
    return preds


def group_dataset_by_lemmas(dataset_file_path):
    """ Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""
    train_morph_list = read_morph_file(dataset_file_path)
    # inflections = collections.OrderedDict()
    inflections = {}
    for lemma, target, feature in train_morph_list:
        # Get forms for lemma (empty set if not exists yet)
        forms = inflections.get(lemma, set())
        # Add new form
        forms.add((target, feature))
        inflections[lemma] = forms
    return inflections


def convert_format(train_file_path, dev_file_path, test_file_path, v_test_file_path, out_file_path):
    for i in [train_file_path, dev_file_path, test_file_path, v_test_file_path, out_file_path]:
        print(i, os.path.exists(i))
    # Read all lemmas from train, dev and test
    train_inflections = group_dataset_by_lemmas(train_file_path)
    dev_inflections = group_dataset_by_lemmas(dev_file_path)
    test_inflections = group_dataset_by_lemmas(test_file_path)
    # Update all lemmas to one dictionary
    for lemma, dev_forms in dev_inflections.items():
        # Get forms for lemma (empty set if not exists yet)
        train_forms = train_inflections.get(lemma, set())
        # Add new form
        train_forms.update(dev_forms)
        train_inflections[lemma] = train_forms
    for lemma, test_forms in test_inflections.items():
        # Get forms for lemma (empty set if not exists yet)
        train_forms = train_inflections.get(lemma, set())
        # Add new form
        train_forms.update(test_forms)
        train_inflections[lemma] = train_forms
    full_paradigms = train_inflections
    # print(train_inflections.items())
    # print(dev_inflections.items())
    # print(test_inflections.items())
    lemmas = read_test_file(v_test_file_path)
    with open(out_file_path, "w", encoding='utf-8') as fp:
        for lemma in lemmas:
            paradigm = full_paradigms.get(lemma)
            for form in paradigm:
                print(lemma, form[0], form[1], sep='\t', file=fp)

if __name__ == '__main__':
    for subb_num in range(1, 4):
        for lang in TASK2_LANGUAGES:
            if os.path.exists(f"task2-data/data/test_langs/{lang}.V-test"):
                v_test_file_path = f"task2-data/data/test_langs/{lang}.V-test"
            else:
                v_test_file_path = f"task2-data/data/dev_langs/{lang}.V-dev"
            convert_format(f"{args.data_dir}/{lang}/uzh.train", f"{args.data_dir}/{lang}/uzh.dev",
                           f"{args.pred_dir}/{subb_num}/test/{lang}.hyp", v_test_file_path, f"task2-out/{subb_num}/{lang}.out")