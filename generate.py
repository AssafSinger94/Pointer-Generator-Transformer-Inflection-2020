import argparse
import os
import torch
import torch.nn.functional as F

import dataset
import data
from data import DATA_FOLDER
import tokenizer
import transformer

# Arguments
parser = argparse.ArgumentParser(description='Evaluating the transformer over test and validation sets')
parser.add_argument('--model-checkpoint', type=str, default='checkpoints/model_best.pth', metavar='M',
                    help="the model file to be evaluated. Usually is of the form model_X.pth(must include folder path)")
parser.add_argument('--valid-file', type=str, default='data', metavar='S',
                    help="Validation file of the dataset (File is located in DATA_FOLDER)")
parser.add_argument('--test-file', type=str, default='data', metavar='S',
                    help="Test file of the dataset (File is located in DATA_FOLDER)")
parser.add_argument('--vocab-file', type=str, default='data', metavar='S',
                    help="Base name of vocabulary files (must include folder path)")
parser.add_argument('--out-file', type=str, default='pred', metavar='D',
                    help="Name of output file containing predictions of the test set (must include folder path)")
args = parser.parse_args()

""" FILES AND TOKENIZER """
# Get validation and test file path
valid_file_path = os.path.join(DATA_FOLDER, args.valid_file)
test_file_path = os.path.join(DATA_FOLDER, args.test_file)
out_file_path = os.path.join(args.out_file)
# Get vocabulary paths
input_vocab_file_path = os.path.join(args.vocab_file + "-input")
output_vocab_file_path = os.path.join(args.vocab_file + "-output")
# Initialize Tokenizer object with input and output vocabulary files
myTokenizer = tokenizer.Tokenizer(input_vocab_file_path, output_vocab_file_path)

""" CONSTANTS """
MAX_SRC_SEQ_LEN = 30
MAX_TGT_SEQ_LEN = 25

""" MODEL AND DATA LOADER """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load model from checkpoint in evaluation mode
model = torch.load(args.model_checkpoint)
model.eval()
# Initialize DataLoader object
# data_loader = dataset.DataLoader(myTokenizer, train_file_path=None, valid_file_path=valid_file_path,
#                                  test_file_path=test_file_path, device=device)
data_loader = dataset.DataLoader(myTokenizer, train_file_path=None, valid_file_path=None,
                                 test_file_path=test_file_path, device=device)


""" FUNCTIONS """
def prdeict_word(data, max_seq_len):
    # Add batch dimension
    data = data.unsqueeze(dim=0)
    outputs = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
    outputs[0] = myTokenizer.sos_id
    for j in range(1, max_seq_len):
        # Compute output of model
        out = model(data, outputs[:, :j]).squeeze()
        # out = F.softmax(out, dim=-1)
        val, ix = out.topk(1)
        outputs[0, j] = ix[-1]
        if ix[-1] == myTokenizer.eos_id:
            break
    return outputs[0, :j + 1]


def write_predictions_to_file(predictions, out_file):
    # Get original input from test file
    lemmas, features = data.read_test_file(test_file_path)
    # Write all data with predictions to the out file
    data.write_morph_file(lemmas, predictions, features, out_file)


def evaluate_validation(max_seq_len=MAX_TGT_SEQ_LEN):
    """ Runs evaluation and computes accuracy over the validation set.
        Predictions generated in sequential order. """
    input_ids, target_ids = data_loader.get_validation_set()
    correct = 0
    # Go over each example
    for i, (data, target) in enumerate(zip(input_ids, target_ids)):
        # Get prediction from model
        pred = prdeict_word(data, max_seq_len)
        if torch.equal(pred, target) and torch.all(torch.eq(pred, target)):
            correct += 1
    print("Validation set: accuracy: %.3f" % (100 * correct / len(input_ids)))


def evaluate(max_seq_len=MAX_TGT_SEQ_LEN):
    """ Runs evaluation over the test set and prints output to prediction file.
        Predictions generated in sequential order. """
    input_ids, input_tokens = data_loader.get_test_set()
    predictions = []
    # Go over each example
    for i, (data, data_tokens) in enumerate(zip(input_ids, input_tokens)):
        # Get prediction from model
        pred = prdeict_word(data, max_seq_len)
        # Strip off sos and eos tokens, and convert from predicted ids to the predicted word
        pred_tokens = myTokenizer.convert_output_ids_to_tokens(pred[1:-1].tolist())

        # where token is unkown token, copy from the source at the same token location
        for j in range(len(pred_tokens)):
            if pred_tokens[j] == myTokenizer.unk and (j < len(data_tokens) - 1):
                pred_tokens[j] = data_tokens[j + 1] # account for data token padded with <s> at the beggining

        pred_word = ''.join(pred_tokens)
        predictions.append(pred_word)
    write_predictions_to_file(predictions, out_file_path)


if __name__ == '__main__':
    # Evaluate accuracy over validation set
    # evaluate_validation()
    # Generating predictions for test set
    evaluate()

# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# valid_file_path = os.path.join(__location__, DATA_FOLDER, args.valid_file)
# test_file_path = os.path.join(__location__, DATA_FOLDER, args.test_file)
# out_file_path = os.path.join(__location__, DATA_FOLDER, args.out_file)
# # Get vocabulary paths
# input_vocab_file_path = os.path.join(__location__, DATA_FOLDER, args.vocab_file + "-input")
# output_vocab_file_path = os.path.join(__location__, DATA_FOLDER, args.vocab_file + "-output")
