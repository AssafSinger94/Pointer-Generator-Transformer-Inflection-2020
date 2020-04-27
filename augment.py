import argparse
from collections import OrderedDict

import data
import utils

parser = argparse.ArgumentParser(description='Reads conll dataset file, and augments the dataset')
parser.add_argument('--src', type=str, default='train',
                    help="Source file of the dataset (must include folder path)")
parser.add_argument('--out', type=str, default='train',
                    help="Output destination for augmented dataset file (must include folder path)")
args = parser.parse_args()

def group_file_by_lemmas(dataset_file):
    """ Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""
    inflections = OrderedDict()
    train_morph_list = data.read_morph_file(dataset_file)
    for lemma, target, feature in train_morph_list:
        # Get forms for lemma (empty set if not exists yet)
        forms = inflections.get(lemma, set())
        # Add new form
        forms.add((target, feature))
        inflections[lemma] = forms
    for lemma, forms in inflections.items():
        forms.add((lemma, 'LEMMA'))
    return inflections

def augment_dataset(dataset_file, out_file):
    utils.maybe_mkdir(out_file)
    aug_file = open(out_file, "w", encoding='utf-8')  # "ISO-8859-1")
    inflections = group_file_by_lemmas(dataset_file)
    # Go over all inflection sets - (word, features)
    for forms in inflections.values():
        for src_pair in forms:
            for tgt_pair in forms:
                # Don't map pair to itself
                if src_pair != tgt_pair:
                    lemma = src_pair[0]
                    target, feature = tgt_pair
                    aug_file.write(f"{lemma}\t{target}\t{feature}\n")
    aug_file.close()


if __name__ == '__main__':
    # Create vocab files
    augment_dataset(args.src, args.out)
