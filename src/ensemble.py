import argparse
import collections
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Gets list of prediction files, and outputs majority voting ensemble '
                                             'pred file')
# parser.add_argument('--pred-list', nargs='+', help='<Required> Set flag', required=True)
parser.add_argument('--lang', type=str, default='train',
                    help="language of the dataset file")
parser.add_argument('--checkpoint-dir', type=str, help="dir of models checkpoints")
args = parser.parse_args()

DATASET_MODES = ["dev", "tst"]
TASK2_LANGUAGES = ['Basque', 'Bulgarian', 'English', 'Finnish', 'German', 'Kannada', 'Maltese',
                   'Navajo', 'Persian', 'Portuguese', 'Russian', 'Spanish', 'Swedish', 'Turkish']

DOWNSAMPLE_LANGUAGES = ['mlg', 'ceb', 'hil', 'mao', 'gmh', 'kon', 'lin', 'sot', 'zul', 'gaa',
 'zpv', 'izh', 'mwf', 'tel', 'gml', 'tgk', 'dje', 'xno', 'kjh', 'lud', 'vro']

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
    """ Gets list of prediction files, and outputs majority voting ensemble pred to output_file"""
    # Get test examples from first file in list
    test_examples = read_morph_file(prediction_files_list[0])
    test_examples_count = len(test_examples)
    # Get only predictions/targets from all files
    predictions_all_files = [read_pred_file(pred_file_path) for pred_file_path in prediction_files_list]
    with open(output_file_path, 'w', encoding='utf-8') as fp:
        # For each example in prediction file
        for i in range(test_examples_count):
            lemma, _, feature = test_examples[i]
            # Get predictions for example i in all files
            preds_for_example = collections.Counter([prediction_list[i] for prediction_list in predictions_all_files])
            # print(preds_for_example)
            majority_vote = preds_for_example.most_common(1)
            # print(majority_vote)
            majority_pred = majority_vote[0][0]
            print(lemma, majority_pred, feature, sep='\t', file=fp)
    print(f"Finish writing out file: {output_file_path}")


# # ------Task 0 - low resource settings - ensembling trm, and pg --------
if __name__ == '__main__':
    ens_num = 1
    for mode in ["dev", "tst"]:
        # Make ensemble dir if not exists
        ensemble_dir = f"ensembles-task0-low/{ens_num}/{mode}"
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        for lang in sorted(DOWNSAMPLE_LANGUAGES):
            ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
            # Get best model of 4 with last 5 epochs
            pred_list = [f"results-task0-low/transformer-base-sml/{mode}/{i}/{lang}.hyp" for i in range(1, 6)]
            pred_list = [pred for pred in pred_list if os.path.exists(pred)]
            print(pred_list)
            ensemble_predictions(pred_list, ensemble_out_file_path)
    ens_num = 2
    for mode in ["dev", "tst"]:
        # Make ensemble dir if not exists
        ensemble_dir = f"ensembles-task0-low/{ens_num}/{mode}"
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)
        for lang in sorted(DOWNSAMPLE_LANGUAGES):
            ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
            # Get best model of 4 with last 5 epochs
            pred_list = [f"results-task0-low/transformer-pg-sml/{mode}/{i}/{lang}.hyp" for i in range(1, 6)]
            print(pred_list)
            pred_list = [pred for pred in pred_list if os.path.exists(pred)]
            print(pred_list)
            ensemble_predictions(pred_list, ensemble_out_file_path)

# # # ------Task 2 - ensembling trm, pg, and trm+pg --------
# if __name__ == '__main__':
#     models = os.listdir(args.checkpoint_dir)
#     ens_num = 1
#     for mode in ["dev", "test"]:
#         # Make ensemble dir if not exists
#         ensemble_dir = f"ensembles-task2/{ens_num}/{mode}"
#         if not os.path.exists(ensemble_dir):
#             os.makedirs(ensemble_dir)
#         for lang in TASK2_LANGUAGES:
#             ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
#             # Get best model of 4 with last 5 epochs
#             pred_list = [f"results-task2/transformer/{mode}/{i}/{lang}.hyp" for i in range(1, 6)] + \
#                         [f"results-task2/transformer-small/{mode}/{i}/{lang}.hyp" for i in range(1, 6)] + \
#                         [f"results-task2/transformer-pg/{mode}/{i}/{lang}.hyp" for i in range(1, 6)] + \
#                         [f"results-task2/transformer-small-pg/{mode}/{i}/{lang}.hyp" for i in range(1, 6)]
#             pred_list = [pred for pred in pred_list if os.path.exists(pred)]
#             print(pred_list)
#             ensemble_predictions(pred_list, ensemble_out_file_path)
#     ens_num = 2
#     for mode in ["dev", "test"]:
#         # Make ensemble dir if not exists
#         ensemble_dir = f"ensembles-task2/{ens_num}/{mode}"
#         if not os.path.exists(ensemble_dir):
#             os.makedirs(ensemble_dir)
#         for lang in TASK2_LANGUAGES:
#             ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
#             # Get best model of 4 with last 5 epochs
#             pred_list = [f"results-task2/transformer/{mode}/{i}/{lang}.hyp" for i in range(1, 6)] + \
#                         [f"results-task2/transformer-small/{mode}/{i}/{lang}.hyp" for i in range(1, 6)]
#             pred_list = [pred for pred in pred_list if os.path.exists(pred)]
#             print(pred_list)
#             ensemble_predictions(pred_list, ensemble_out_file_path)
#     ens_num = 3
#     for mode in ["dev", "test"]:
#         # Make ensemble dir if not exists
#         ensemble_dir = f"ensembles-task2/{ens_num}/{mode}"
#         if not os.path.exists(ensemble_dir):
#             os.makedirs(ensemble_dir)
#         for lang in TASK2_LANGUAGES:
#             ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
#             # Get best model of 4 with last 5 epochs
#             pred_list = [f"results-task2/transformer-pg/{mode}/{i}/{lang}.hyp" for i in range(1, 6)] + \
#                         [f"results-task2/transformer-small-pg/{mode}/{i}/{lang}.hyp" for i in range(1, 6)]
#             pred_list = [pred for pred in pred_list if os.path.exists(pred)]
#             print(pred_list)
#             ensemble_predictions(pred_list, ensemble_out_file_path)

# # ------Task 0 - ensembleing model 5 and five last epochs--------
# if __name__ == '__main__':
#     lang_dirs = os.listdir(args.checkpoint_dir)
#     for mode in DATASET_MODES:
#         ens_num = 2
#         # Make ensemble dir if not exists
#         ensemble_dir = f"ensembles/{mode}/{ens_num}"
#         if not os.path.exists(ensemble_dir):
#             os.makedirs(ensemble_dir)
#         for lang in lang_dirs:
#             ensemble_out_file_path = f"{ensemble_dir}/{lang}.hyp"
#             # Get best model of 4 with last 5 epochs
#             pred_list = [f"results/{mode}/4/{lang}.hyp"] + [f"results/{mode}/4_{i}/{lang}.hyp" for i in range(1, 6)]
#             pred_list = [pred for pred in pred_list if os.path.exists(pred)]
#             print(pred_list)
#             ensemble_predictions(pred_list, ensemble_out_file_path)


# # ------Task 0 - ensembleing models 1, 2, 4--------
# if __name__ == '__main__':
#     lang_dirs = os.listdir(args.checkpoint_dir)
#     for mode in DATASET_MODES:
#         ens_num = 1
#         # Make ensemble dir if not exists
#         if not os.path.exists(f"ensembles/{mode}/{ens_num}"):
#             os.makedirs(f"ensembles/{mode}/{ens_num}")
#         for lang in lang_dirs:
#             pred_list = [f"results/{mode}/{i}/{lang}.hyp" for i in [1, 2, 4]]
#             print(pred_list)
#             stop =False
#             for pred in pred_list:
#                 if not os.path.exists(pred):
#                     stop = True
#                     if os.path.exists(pred_list[2]):
#                         print(f"{pred} not exists, just using model 4")
#                         shutil.copy(pred_list[2], f"ensembles/{mode}/{ens_num}/{lang}.hyp")
#                     else:
#                         print(f"{pred} not exists, just using model 1")
#                         shutil.copy(pred_list[0], f"ensembles/{mode}/{ens_num}/{lang}.hyp")
#             if not stop:
#                 # Create ensemble file
#                 ensemble_predictions(pred_list, f"ensembles/{mode}/{ens_num}/{lang}.hyp")
#
# ensemble_predictions(args.pred_list, f"ensembles/{args.ens_num}/{args.lang}.hyp")
# pred_list = [f"{args.checkpoint_dir}/{lang}/aug/{i}/{lang}.dev.pred" for i in [1, 2, 4]]