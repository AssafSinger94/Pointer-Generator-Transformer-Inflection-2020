import argparse
import os
import re
import shutil

parser = argparse.ArgumentParser(description='Reads conll dev file, and covers the target (test file format)')
parser.add_argument('--checkpoint-dir', type=str, default='train',
                    help="Folder of the checkpoint files")
parser.add_argument('--data-dir', type=str, default='task0-data',
                    help="Folder of the dataset file")
parser.add_argument('--lang', type=str, default='train',
                    help="language of the dataset file")
args = parser.parse_args()

DEV_LANGUAGES = ['aka', 'ang', 'azg', 'ceb', 'cly', 'cpa', 'ctp', 'czn', 'dan', 'deu', 'eng', 'est',
                 'fin', 'frr', 'gaa', 'gmh', 'hil', 'isl', 'izh', 'kon', 'krl', 'lin', 'liv', 'lug',
                 'mao', 'mdf', 'mhr', 'mlg', 'myv', 'nld', 'nob', 'nya', 'ote', 'otm', 'pei', 'sme',
                 'sot', 'swa', 'swe', 'tgl', 'vep', 'vot', 'xty', 'zpv', 'zul']

def rename_files(checkpoint_dir, language, dataset_type):
    # get all files in checkpoint dir
    if os.path.exists(f"{checkpoint_dir}/{dataset_type}/1"):
        checkpoint_dir = f"{checkpoint_dir}/{dataset_type}/1"
    else:
        return
    checkpoint_files = os.listdir(checkpoint_dir)
    for filename in checkpoint_files:
        print(f"File name: {checkpoint_dir}/{filename}")
        if filename[0] == ".":
            pass
            # new_filename = f"model-{language}{filename}"
            # print(f"Changed to: {checkpoint_dir}/{new_filename}")
            # os.rename(f"{checkpoint_dir}/{filename}", f"{checkpoint_dir}/{new_filename}")


def get_accuracy(filename):
    # string = "model-lug.nll_0.6892.acc_92.8425.dist_1.epoch_25"
    result = re.search('.acc_(.*).dist', filename)
    return float(result.group(1))


def get_epoch(filename):
    result = re.search('.epoch_(.*)', filename)
    return int(result.group(1))


def keep_only_best_checkpoint(checkpoint_dir, language, dataset_type):
    # get all files in checkpoint dir
    if os.path.exists(f"{checkpoint_dir}/{dataset_type}/1"):
        checkpoint_dir = f"{checkpoint_dir}/{dataset_type}/1"
    else:
        return
    checkpoint_files = os.listdir(checkpoint_dir)
    best_acc = -1
    best_acc_filename = ""
    for filename in checkpoint_files:
        if filename.startswith(f"model-{language}.nll"):
            curr_acc = get_accuracy(filename)
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_acc_filename = filename
    print(f"Best checkpoint: {checkpoint_dir}/{best_acc_filename} - acc: {best_acc}")
    # Iterate again over all files
    for filename in checkpoint_files:
        if filename.startswith(f"model-{language}.nll"):
            if filename != best_acc_filename:
                os.remove(f"{checkpoint_dir}/{filename}")
    if os.path.exists(f"{checkpoint_dir}/model-{language}.progress"):
        os.remove(f"{checkpoint_dir}/model-{language}.progress")
    if os.path.exists(f"{checkpoint_dir}/.log"):
        os.remove(f"{checkpoint_dir}/.log")


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
                shutil.copy(f"{curr_checkpoint_dir}/{best_acc_filename}", f"{curr_checkpoint_dir}/model_best.epoch_0")
                print(f"copied {curr_checkpoint_dir}/{best_acc_filename} to {curr_checkpoint_dir}/model_best.epoch_0")


def duplicate_checkpoints(checkpoint_dir, language):
    for i in range(6, 7):
        if os.path.exists(f"{checkpoint_dir}/aug/{i}"):
            curr_checkpoint_dir = f"{checkpoint_dir}/aug/{i}"
            # get all files in checkpoint dir
            checkpoint_files = os.listdir(curr_checkpoint_dir)
            count = 1
            for filename in checkpoint_files:
                if filename.startswith(f"model-{language}.nll"):
                    shutil.copy(f"{curr_checkpoint_dir}/{filename}", f"{curr_checkpoint_dir}/model-{count}.epoch_0")
                    print(f"copied {curr_checkpoint_dir}/{filename} to {curr_checkpoint_dir}/model-{count}.epoch_0")
                    count += 1


def transfer_pretrained_model(checkpoint_dir, language):
    if os.path.exists(f"{checkpoint_dir}/hall/1"):
        checkpoint_files = os.listdir(f"{checkpoint_dir}/hall/1")
        for filename in checkpoint_files:
            if filename.startswith(f"model-{language}.nll"):
                for i in range(1, 7):
                    dest_dir = f"{checkpoint_dir}/aug/{i}"
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)
                    if not os.path.exists(f"{dest_dir}/pretrained_model.epoch_0"):
                        shutil.copyfile(f"{checkpoint_dir}/hall/1/{filename}", f"{dest_dir}/pretrained_model.epoch_0")
                        print(f"moved {checkpoint_dir}/hall/1/{filename} to {dest_dir}/pretrained_model.epoch_0")


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


def read_baseline_pred_file(baseline_prediction_file_path):
    # Get all lines in file
    base_pred_file = open(baseline_prediction_file_path, 'r', encoding='utf-8')
    lines = base_pred_file.readlines()
    preds = []
    for line in lines:
        if not line or line == "\n" or line.startswith("prediction"):
            continue
        # Strip '\n' and split, add only first component (predictions)
        preds.append(line.replace("\n", "").split("\t")[0])
    base_pred_file.close()
    return preds


def process_generated_file_to_output_file_epoch_runs(data_dir, checkpoint_dir, language, dataset_type, dataset_mode):
    mode_to_mode = {"tst": "test", "dev": "dev"}
    for i in range(1, 6):
        if os.path.exists(f"{checkpoint_dir}/{dataset_type}/4/epoch-{i}"):
            # Get prediction file produced by baseline code
            curr_checkpoint_dir = f"{checkpoint_dir}/{dataset_type}/4/epoch-{i}"
            baseline_prediction_file_path = os.path.join(curr_checkpoint_dir,
                                                         f"model-{language}.decode.{mode_to_mode.get(dataset_mode)}.tsv")
            # Get test file
            test_file_path = os.path.join(data_dir, f"{language}.{dataset_mode}")
            output_file_path = os.path.join(curr_checkpoint_dir, f"{language}.{dataset_mode}.pred")
            # If no pred file, skip
            if not os.path.exists(baseline_prediction_file_path):
                print(f"File not exists: {baseline_prediction_file_path}")
                continue
            # Get list of test examples and predictions
            test_examples = read_morph_file(test_file_path)
            predictions = read_baseline_pred_file(baseline_prediction_file_path)
            print(f"test: {test_file_path}")
            print(f"base pred: {baseline_prediction_file_path}")
            print(len(test_examples), len(predictions))
            if len(test_examples) != len(predictions):
                print("test set not ready")
                continue
            print(test_examples[0], predictions[0])
            with open(output_file_path, 'w', encoding='utf-8') as fp:
                for (lemma, _, feature), pred in zip(test_examples, predictions):
                    print(lemma, pred, feature, sep='\t', file=fp)
            print(f"Finish writing out file: {output_file_path}")
            if not os.path.exists(f"results/{dataset_mode}/4_{i}/"):
                os.makedirs(f"results/{dataset_mode}/4_{i}")
            print(f" Copied to results/{dataset_mode}/4_{i}/{language}.hyp")
            shutil.copy(output_file_path, f"results/{dataset_mode}/4_{i}/{language}.hyp")

def process_generated_file_to_output_file(data_dir, checkpoint_dir, results_dir, language, dataset_type, dataset_mode):
    mode_to_mode = {"tst": "test", "dev": "dev"}
    for i in range(1, 2):
        if os.path.exists(f"{checkpoint_dir}/{dataset_type}/{i}"):
            # Get prediction file produced by baseline code
            curr_checkpoint_dir = f"{checkpoint_dir}/{dataset_type}/{i}"
            baseline_prediction_file_path = os.path.join(curr_checkpoint_dir,
                                                         f"model-{language}.decode.{mode_to_mode.get(dataset_mode)}.tsv")
            test_file_path = os.path.join(data_dir, f"{language}.{dataset_mode}")
            # baseline_prediction_file_path = os.path.join(curr_checkpoint_dir,
            #                                              f"model-{language}.decode.{dataset_mode}.tsv")
            # # Get test file
            # test_file_path = os.path.join(data_dir, f"uzh.{dataset_mode}")
            output_file_path = os.path.join(curr_checkpoint_dir, f"{language}.{dataset_mode}.pred")
            # If no pred file, skip
            print(f"test: {test_file_path}")
            print(f"base pred: {baseline_prediction_file_path}")
            if not os.path.exists(baseline_prediction_file_path):
                print("No prediction file")
                continue
            # Get list of test examples and predictions
            test_examples = read_morph_file(test_file_path)
            predictions = read_baseline_pred_file(baseline_prediction_file_path)
            print(len(test_examples), len(predictions))
            if len(test_examples) != len(predictions):
                print("test set not ready")
                continue
            # print(test_examples[0], predictions[0])
            with open(output_file_path, 'w', encoding='utf-8') as fp:
                for (lemma, _, feature), pred in zip(test_examples, predictions):
                    print(lemma, pred, feature, sep='\t', file=fp)
            print(f"Finish writing out file: {output_file_path}")
            if not os.path.exists(f"{results_dir}/{dataset_mode}/{i}/"):
                os.makedirs(f"{results_dir}/{dataset_mode}/{i}")
            print(f" Copied to {results_dir}/{dataset_mode}/{i}/{language}.hyp")
            shutil.copy(output_file_path, f"{results_dir}/{dataset_mode}/{i}/{language}.hyp")

# ------Task 0 - Low resource-------
if __name__ == '__main__':
    for model in ["transformer-base", "transformer-pg"]:
        for data_type in ["trn", "aug"]:
            print(f"Model: {model} {data_type}")
            model_results_dir = f"ablation-studies/{model}/{data_type}"
            lang_dirs = sorted(os.listdir(f"{args.checkpoint_dir}/{model}"))
            for lang in lang_dirs:
                model_lang_checkpoints_dir = f"{args.checkpoint_dir}/{model}/{lang}"
                process_generated_file_to_output_file(args.data_dir, model_lang_checkpoints_dir,
                                                      model_results_dir, lang, data_type, "dev")
                process_generated_file_to_output_file(args.data_dir, model_lang_checkpoints_dir,
                                                      model_results_dir, lang, data_type, "tst")

# # ------ Task 0 - General -------
#     lang_dirs = os.listdir(args.checkpoint_dir)
#     for lang in lang_dirs:
#         process_generated_file_to_output_file(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug",
#                                                          "dev")
#         process_generated_file_to_output_file(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug",
#                                                          "tst")




# # ------Task 0 - Low resource-------
# if __name__ == '__main__':
#     models = os.listdir(args.checkpoint_dir)
#     for model in models:
#         print(f"Model: {model}")
#         model_results_dir = f"results-task0-low/{model}-sml"
#         lang_dirs = sorted(os.listdir(f"{args.checkpoint_dir}/{model}"))
#         for lang in lang_dirs:
#             model_lang_checkpoints_dir = f"{args.checkpoint_dir}/{model}/{lang}"
#             process_generated_file_to_output_file(args.data_dir, model_lang_checkpoints_dir,
#                                                   model_results_dir, lang, "sml", "dev")
#             process_generated_file_to_output_file(args.data_dir, model_lang_checkpoints_dir,
#                                                   model_results_dir, lang, "sml", "tst")

# # ------ Task 2 -------
# if __name__ == '__main__':
#     models = os.listdir(args.checkpoint_dir)
#     for model in models:
#         print(f"Model: {model}")
#         lang_dirs = sorted(os.listdir(f"{args.checkpoint_dir}/{model}"))
#         for lang in lang_dirs:
#             process_generated_file_to_output_file(f"{args.data_dir}/{lang}",
#                                                   f"{args.checkpoint_dir}/{model}/{lang}",
#                                                   f"results-task2/{model}", lang, "trn", "dev")
#             process_generated_file_to_output_file(f"{args.data_dir}/{lang}",
#                                                   f"{args.checkpoint_dir}/{model}/{lang}",
#                                                   f"results-task2/{model}", lang, "trn", "test")
# # ------ Task 0 - General -------
    # lang_dirs = os.listdir(args.checkpoint_dir)
    # for lang in lang_dirs:
    #     # rename_files(f"{args.checkpoint_dir}/{lang}", lang, "aug")
    #     # keep_only_best_checkpoint(f"{args.checkpoint_dir}/{lang}", lang, "hall")
    #     # transfer_pretrained_model(f"{args.checkpoint_dir}/{lang}", lang)
    #     # best_checkpoint_duplicate(f"{args.checkpoint_dir}/{lang}", lang, "aug")
    #     # check_predicted_files(f"{args.checkpoint_dir}/{lang}", lang)
    #     duplicate_checkpoints(f"{args.checkpoint_dir}/{lang}", lang)
    #     # process_generated_file_to_output_file_epoch_runs(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug", "dev")
    #     # process_generated_file_to_output_file_epoch_runs(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug", "tst")
    #
    #     process_generated_file_to_output_file(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug",
    #                                                      "dev")
    #     process_generated_file_to_output_file(args.data_dir, f"{args.checkpoint_dir}/{lang}", lang, "aug",
    #                                                      "tst")


# def check_predicted_files(checkpoint_dir, language):
#     dataset_type = "aug"
#     mode_to_mode = {"tst": "test", "dev": "dev"}
#     for dataset_mode in ["tst", "dev"]:
#         for i in range(1, 3):
#             if os.path.exists(f"{checkpoint_dir}/{dataset_type}/{i}/model-{language}.decode.{mode_to_mode.get(dataset_mode)}.tsv"):
#                 print(f"File exists {checkpoint_dir}/{dataset_type}/{i}/model-{language}.decode.{mode_to_mode.get(dataset_mode)}.tsv")
