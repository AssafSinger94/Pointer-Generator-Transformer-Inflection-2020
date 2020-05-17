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

ALL_LANGUAGES = []

ENSEMBLE = "ens"
REGULAR = "reg"

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

def evaluate_predictions(langs, pred_type, model_num):
    preds_dir = "ensembles/dev" if pred_type == ENSEMBLE else "results/dev"
    for lang in langs:
        if lang.startswith("dropout"):
            continue
        print(lang + "-")
        os.system(f"python task0-data/evaluate.py --ref task0-data/out/{lang}.dev"
                  f" --hyp {preds_dir}/{model_num}/{lang}.hyp")

if __name__ == '__main__':
    # collect_model_results(args.checkpoint_dir)
    langs = os.listdir(args.checkpoint_dir)
    # Get results for all languages
    # evaluate_predictions(langs, ENSEMBLE, "1")
    # evaluate_predictions(langs, REGULAR, "4")
    # # Get results for dev languages
    # evaluate_predictions(DEV_LANGUAGES, ENSEMBLE, "1")
    # evaluate_predictions(DEV_LANGUAGES, REGULAR, "4")
    # Get results for ensemble 2
    # evaluate_predictions(langs, ENSEMBLE, "2")
    # evaluate_predictions(DEV_LANGUAGES, ENSEMBLE, "2")

    evaluate_predictions(langs, REGULAR, "6")
    # Get results for dev languages
    evaluate_predictions(DEV_LANGUAGES, REGULAR, "6")

    # for lang in langs:
    #     if not os.path.exists(
    #             f"checkpoints/sigmorphon20-task0/transformer/{lang}/aug/6/model-{lang}.decode.test.tsv"):
    #         print(
    #             f"Not exists checkpoints/sigmorphon20-task0/transformer/{lang}/aug/6/model-{lang}.decode.test.tsv")
    # for lang in langs:
    #     if lang.startswith("dropout"):
    #         continue
    #     for i in range(1, 6):
    #         if not os.path.exists(f"checkpoints/sigmorphon20-task0/transformer/{lang}/aug/4/epoch-{i}/model-{lang}.decode.test.tsv"):
    #             print(f"Not exists checkpoints/sigmorphon20-task0/transformer/{lang}/aug/4/epoch-{i}/model-{lang}.decode.test.tsv")


# string = "model-lug.nll_0.6892.acc_92.8425.dist_1.epoch_25"