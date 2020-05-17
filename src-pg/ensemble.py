import argparse
import collections
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Reads conll dev file, and covers the target (test file format)')
parser.add_argument('--pred-list', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--lang', type=str, default='train',
                    help="language of the dataset file")
parser.add_argument('--ens-num', type=str, help="ensemble number")
args = parser.parse_args()

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

def read_pred_file(pred_file_path):
    """ Reads conll file, Returns only targets"""
    # Get all lines in file
    base_pred_file = open(pred_file_path, 'r', encoding='utf-8')
    lines = base_pred_file.readlines()
    preds = []
    for line in lines:
        if not line or line == "\n" or line.startswith("prediction"):
            continue
        # Strip '\n' and split, add only first component (predictions)
        preds.append(line.replace("\n", "").split("\t")[1])
    base_pred_file.close()
    return preds

def ensemble_predictions(prediction_files_list, output_file_path):
    # Get test examples from first file in list
    test_examples = read_morph_file(prediction_files_list[0])
    test_examples_count = len(test_examples)
    # Get only predictions/targets from all files
    predictions_all_files = [read_pred_file(pred_file_path) for pred_file_path in prediction_files_list]
    with open(output_file_path, 'w+', encoding='utf-8') as fp:
        # For each example in prediction file
        for i in range(test_examples_count):
            lemma, _, feature = test_examples[i]
            # Get predictions for example i in all files
            preds_for_example = collections.Counter([prediction_list[i] for prediction_list in predictions_all_files])
            print(preds_for_example)
            majority_vote = preds_for_example.most_common(1)
            print(majority_vote)
            majority_pred = majority_vote[0][0]
            print(lemma, majority_pred, feature, sep='\t', file=fp)
    print(f"Finish writing out file: {output_file_path}")


if __name__ == '__main__':
    print(args.pred_list)
    if not os.path.exists(f"ensembles/{args.ens_num}"):
        os.makedirs(f"ensembles/{args.ens_num}")
    ensemble_predictions(args.pred_list, f"ensembles/{args.ens_num}/{args.lang}.hyp")
    # lang_dirs = os.listdir(args.checkpoint_dir)
    # for lang in lang_dirs:
    #     pass
