import argparse
import os

import data
from data import DATA_FOLDER

# Arguments
parser = argparse.ArgumentParser(description='Computing accuracy of model predictions compare to target file')
parser.add_argument('--pred-file', type=str, default='data', metavar='S',
                    help="File with model predictions")
parser.add_argument('--target-file', type=str, default='data', metavar='S',
                    help="File with gold targets")
args = parser.parse_args()

""" Files """
# Get validation and test file path
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
pred_file_path = os.path.join(__location__, DATA_FOLDER, args.pred_file)
target_file_path = os.path.join(__location__, DATA_FOLDER, args.target_file)

""" FUNCTIONS """
def accuracy(predictions, targets):
    """Return fraction of matches between two lists sequentially."""
    correct_count = 0
    for prediction, target in zip(predictions, targets):
        if prediction == target:
            correct_count += 1
    return correct_count / len(predictions)

"""Version with words NOT cleaned"""
def evaluate_predictions(pred_file, target_file):
    pred_lines = data.read_morph_file(pred_file)
    target_lines = data.read_morph_file(target_file)
    predictions = [line[1] for line in pred_lines]
    truth = [line[1] for line in target_lines]
    print("Test set. accuracy: %.4f" % accuracy(predictions, truth))

"""Version with words cleaned"""
# def evaluate_predictions(pred_file, target_file):
#     _, predictions, _ = data.read_train_file(pred_file)
#     _, truth, _ = data.read_train_file(target_file)
#     print("Test set. accuracy: %.4f" % accuracy(predictions, truth))


if __name__ == '__main__':
    # Compute accuracy of predictions compare to truth
    evaluate_predictions(pred_file_path, target_file_path)