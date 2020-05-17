import argparse
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Reads conll dev file, and covers the target (test file format)')
parser.add_argument('--checkpoint-dir', type=str, default='train',
                    help="Folder of the checkpoint files")
parser.add_argument('--lang', type=str, default='train',
                    help="language of the dataset file")
args = parser.parse_args()

DEV_LANGUAGES = ['aka', 'ang', 'azg', 'ceb', 'cly', 'cpa', 'ctp', 'czn', 'dan', 'deu', 'eng', 'est',
                 'fin', 'frr', 'gaa', 'gmh', 'hil', 'isl', 'izh', 'kon', 'krl', 'lin', 'liv', 'lug',
                 'mao', 'mdf', 'mhr', 'mlg', 'myv', 'nld', 'nob', 'nya', 'ote', 'otm', 'pei', 'sme',
                 'sot', 'swa', 'swe', 'tgl', 'vep', 'vot', 'xty', 'zpv', 'zul']
LOW_RESOURCE_LANGUAGES = ['mlg', 'ceb', 'hil', 'mao', 'gmh', 'kon', 'lin', 'sot', 'zul', 'gaa',
 'zpv', 'izh', 'mwf', 'tel', 'gml', 'tgk', 'dje', 'xno', 'kjh', 'lud', 'vro']
TASK2_LANGUAGES = ['Basque', 'Bulgarian', 'English', 'Finnish', 'German', 'Kannada', 'Maltese',
                   'Navajo', 'Persian', 'Portuguese', 'Russian', 'Spanish', 'Swedish', 'Turkish']

TASK0_DATASET = "task0-data/out"
TASK2_DATASET = "task2-data"

ENSEMBLE = "ens"
REGULAR = "reg"

ENSEMBLE_DIR = "ensembles/dev"
REGULAR_DIR = "results/dev"
LOW_RESOURCE_DIR = "results-task0-low"
LOW_RESOURCE_ENSEMBLE_DIR = "ensembles-task0-low"

def get_accuracy(filename):
    result = re.search('.acc_(.*).dist', filename)
    return float(result.group(1))


def get_epoch(filename):
    result = re.search('.epoch_(.*)', filename)
    return int(result.group(1))


def best_checkpoint_duplicate(checkpoint_dir, language, dataset_type):
    for i in range(1, 3):
        if os.path.exists(f"{checkpoint_dir}/{dataset_type}/{i}"):
            curr_checkpoint_dir = f"{checkpoint_dir}/{dataset_type}/{i}"
            # get all files in checkpoint dir
            checkpoint_files = os.listdir(curr_checkpoint_dir)
            best_acc = -1
            best_acc_filename = ""
            for filename in checkpoint_files:
                if filename.startswith(f"model-{language}.nll"):
                    curr_acc = get_accuracy(filename)
                    if curr_acc > best_acc:
                        best_acc = curr_acc
                        best_acc_filename = filename
            print(f"Best checkpoint: {curr_checkpoint_dir}/{best_acc_filename} - acc: {best_acc}")
            if best_acc >= 0:
                shutil.copy(f"{curr_checkpoint_dir}/{best_acc_filename}", f"{curr_checkpoint_dir}/model_best")
                print(f"copied {curr_checkpoint_dir}/{best_acc_filename} to {curr_checkpoint_dir}/model_best")


def collect_results_for_lang(checkpoint_dir, language):
    results = [''] * 7
    # Get accuracy of hallucinated pretrain
    if os.path.exists(f"{checkpoint_dir}/hall/1"):
        # get all files in checkpoint dir
        checkpoint_files = os.listdir(f"{checkpoint_dir}/hall/1")
        for filename in checkpoint_files:
            if filename.startswith(f"model-{language}.nll"):
                acc = get_accuracy(filename)
                epoch = get_epoch(filename)
                results[0] = str(acc)
    # Get accuracy of different runs
    for i in range(1, 3):
        if os.path.exists(f"{checkpoint_dir}/aug/{i}"):
            # get all files in checkpoint dir
            checkpoint_files = os.listdir(f"{checkpoint_dir}/aug/{i}")
            for filename in checkpoint_files:
                if filename.startswith(f"model-{language}.nll"):
                    acc = get_accuracy(filename)
                    epoch = get_epoch(filename)
                    results[i] = str(acc)
    for i in range(3, 7):
        if os.path.exists(f"{checkpoint_dir}/aug/{i}"):
            # get all files in checkpoint dir
            checkpoint_files = os.listdir(f"{checkpoint_dir}/aug/{i}")
            best_acc = -1
            for filename in checkpoint_files:
                if filename.startswith(f"model-{language}.nll"):
                    curr_acc = get_accuracy(filename)
                    if curr_acc > best_acc:
                        best_acc = curr_acc
            results[i] = str(best_acc)
    return results

def collect_model_results(checkpoint_dir):
    lang_dirs = os.listdir(checkpoint_dir)
    lang_results = {}
    for lang in lang_dirs:
        lang_results[lang] = collect_results_for_lang(f"{checkpoint_dir}/{lang}", lang)
    # Print all once
    for lang, results in lang_results.items():
        print('\t'.join([lang] + results))
    # Second time, print only those in DEV LANG
    print("\n")
    for lang, results in lang_results.items():
        if lang in DEV_LANGUAGES:
            print('\t'.join([lang] + results))


def collect_model_results_task2(checkpoint_dir):
    models = os.listdir(checkpoint_dir)
    # Go over all the models
    for model in models:
        print(f"Model: {model}")
        lang_dirs = sorted(os.listdir(f"{checkpoint_dir}/{model}"))
        lang_results = {}
        # Go over all the languages
        for lang in lang_dirs:
            results = ['XX'] * 5
            # Get best accuracy for each run
            for i in range(1, 6):
                if os.path.exists(f"{checkpoint_dir}/{model}/{lang}/trn/{i}"):
                    # get all files in checkpoint dir
                    checkpoint_files = os.listdir(f"{checkpoint_dir}/{model}/{lang}/trn/{i}")
                    best_acc = -1
                    for filename in checkpoint_files:
                        if filename.startswith(f"model-{lang}.nll"):
                            curr_acc = get_accuracy(filename)
                            if curr_acc > best_acc:
                                best_acc = curr_acc
                    # Sey best_acc in run number
                    results[i - 1] = str(best_acc)
            # Set list of accuracies for each lang
            lang_results[lang] = results
        # print results for all languages
        for lang, results in lang_results.items():
            print('\t'.join([lang] + results))

# def evaluate_predictions(langs, preds_dir, model_num):
#     for lang in langs:
#         if lang.startswith("dropout"):
#             continue
#         print(lang + "-")
#         os.system(f"python task0-data/evaluate.py --ref {TASK0_DATASET}/{lang}.dev"
#                   f" --hyp {preds_dir}/{model_num}/{lang}.hyp")


def evaluate_predictions(langs, preds_dir):
    for lang in langs:
        if lang.startswith("dropout"):
            continue
        print(lang + "-")
        os.system(f"python task0-data/evaluate.py --ref {TASK0_DATASET}/{lang}.dev"
                  f" --hyp {preds_dir}/{lang}.hyp")


# # -----Task0 code - original testing-------
if __name__ == '__main__':
    langs = sorted(os.listdir(args.checkpoint_dir))
    print("All langs. Model 1")
    evaluate_predictions(langs, f"{REGULAR_DIR}/4")
    print("Low resource langs. Model 1")
    # Get results for dev languages
    evaluate_predictions(sorted(LOW_RESOURCE_LANGUAGES), f"{REGULAR_DIR}/4")
    other_langs = sorted([lang for lang in langs if lang not in LOW_RESOURCE_LANGUAGES])
    print("Other langs. Model 1")
    # Get results for dev languages
    evaluate_predictions(other_langs, f"{REGULAR_DIR}/4")



# # -----Task0 code - Ablation Studies-------
# if __name__ == '__main__':
#     langs = sorted(os.listdir(args.checkpoint_dir))
#     ablation_dir = "ablation-studies"
#     for model_number in range(2, 6):
#         print("All langs. Model", model_number)
#         # Get results for dev languages
#         evaluate_predictions(langs, f"{ablation_dir}/{model_number}/dev/1")
#     for model_number in range(2, 6):
#         print("Low resource langs. Model", model_number)
#         # Get results for dev languages
#         evaluate_predictions(sorted(LOW_RESOURCE_LANGUAGES), f"{ablation_dir}/{model_number}/dev/1")
#     other_langs = sorted([lang for lang in langs if lang not in LOW_RESOURCE_LANGUAGES])
#     for model_number in range(2, 6):
#         print("Other langs. Model", model_number)
#         # Get results for dev languages
#         evaluate_predictions(other_langs, f"{ablation_dir}/{model_number}/dev/1")



# -----Task0 code - low resource testing-------
# if __name__ == '__main__':
#     langs = sorted(LOW_RESOURCE_LANGUAGES)
#     evaluate_predictions(langs, f"{LOW_RESOURCE_DIR}/baseline/dev")
    # evaluate_predictions(langs, f"{LOW_RESOURCE_ENSEMBLE_DIR}/1/dev")
    # evaluate_predictions(langs, f"{LOW_RESOURCE_ENSEMBLE_DIR}/2/dev")

# if __name__ == '__main__':
#     langs = sorted(LOW_RESOURCE_LANGUAGES)
#     # Get models with results
#     for model in sorted(os.listdir(LOW_RESOURCE_DIR)):
#         if model in ["transformer-base", "transformer-pg"]:
#             continue
#         print(model)
#         # langs = os.listdir(f"{args.checkpoint_dir}/{model}")
#         evaluate_predictions(langs, f"{LOW_RESOURCE_DIR}/{model}/dev", "1")
#         evaluate_predictions(langs, f"{LOW_RESOURCE_DIR}/{model}/dev", "2")

# # -----Task0 code - original testing-------
# if __name__ == '__main__':
#     model_name = "NYU-CUBoulder-0"
#     langs = sorted(os.listdir(args.checkpoint_dir))
#     # Get results for dev languages
#     # print("NYU-CUBoulder-01")
#     # evaluate_predictions(langs, f"{ENSEMBLE_DIR}/1")
#     # print("NYU-CUBoulder-02")
#     # evaluate_predictions(langs, f"{REGULAR_DIR}/4")
#     # print("NYU-CUBoulder-03")
#     # evaluate_predictions(langs, f"{ENSEMBLE_DIR}/2")
#     print("NYU-CUBoulder-04")
#     evaluate_predictions(langs, f"{REGULAR_DIR}/6")


#     # Get results for dev languages
#     evaluate_predictions(DEV_LANGUAGES, REGULAR_DIR, "6")
#     evaluate_predictions(DEV_LANGUAGES, ENSEMBLE_DIR, "1")

# -----Task2 code-------
# if __name__ == '__main__':
#     collect_model_results_task2(args.checkpoint_dir)
#     # Evaluate predictions of task2 models
#     for model in sorted(os.listdir(LOW_RESOURCE_DIR)):
#         if model == "transformer" or model.startswith("droupout"):
#             continue
#         print(model)
#         # print(os.listdir(f"{args.checkpoint_dir}/{model}"))
#         langs = os.listdir(f"{args.checkpoint_dir}/{model}")
#         evaluate_predictions(langs, f"{LOW_RESOURCE_DIR}/{model}/dev", "1")