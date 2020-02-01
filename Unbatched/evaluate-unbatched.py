import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
import tokenizer
import model

# Training settings
parser = argparse.ArgumentParser(description='Transformer for morphological inflection')
parser.add_argument('--test-file', type=str, default='data', metavar='S',
                    help="Test file of the dataset")
parser.add_argument('--vocab-file', type=str, default='data', metavar='S',
                    help="Base name of vocabulary files")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def load_checkpoint(filepath):
#     checkpoint = torch.load(filepath)
#     model = checkpoint['model']
#     model.load_state_dict(checkpoint['state_dict'])
#     for parameter in model.parameters():
#         parameter.requires_grad = False
#
#     model.eval()
#     return model
#
#
# model = load_checkpoint(args.model)
# model.eval()
# state_dict = torch.load(args.model)
# # ------
# # Model class must be defined somewhere
model = torch.load(args.model)
# ------
# model = state_dict['model']
# # model = nn.model(*args, **kwargs)
# model.load_state_dict(state_dict['state_dict'])
model.eval()


def get_test_dataset(test_file_path, tokenizer):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(test_file_path)
    # Pad target with sos and eos symbols
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Get tensors of all input ids and output ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    output_ids = tokenizer.get_id_tensors(outputs_tokens, device, "OUTPUT")
    return inputs_ids, output_ids


def evaluate(input_ids, target_ids, max_seq_len=30):
    """ Runs full validation epoch over the dataset
	CHECK EACH TO MAKE input_ids, target_ids GLOBAL"""
    # model.eval()
    count = 0
    correct = 0
    # Go over each example
    for i, (data, target) in enumerate(zip(input_ids, target_ids)):
        outputs = torch.zeros(max_seq_len, dtype=torch.long, device=device)
        outputs[0] = tokenizer.convert_output_tokens_to_ids([tokenizer.sos])[0]
        for i in range(1, max_seq_len):
            # Compute output of model
            out = model(data, outputs[:i]).squeeze()
            out = F.softmax(out, dim=-1)
            val, ix = out.topk(1)

            outputs[i] = ix[-1]
            if ix[-1] == tokenizer.convert_output_tokens_to_ids([tokenizer.eos])[0]:
                break
        pred = outputs[:i + 1]
        if torch.equal(pred, target) and torch.all(torch.eq(pred, target)):
            correct += 1
        count += 1
    print("Validation set: accuracy: %.3f" % (100 * correct / count))
    # return ' '.join(tokenizer.convert_output_ids_to_tokens(outputs[:i]))


if __name__ == '__main__':
    # Get location of current folder, to work with full file paths
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    test_file_path = os.path.join(__location__, args.test_file)
    # Get vocabulary paths
    vocab_file_path = os.path.join(__location__, args.vocab_file)
    input_vocab_file_path = vocab_file_path + "-input"
    output_vocab_file_path = vocab_file_path + "-output"
    # Initialize tokenizer object with input and output vocabulary files
    tokenizer = tokenizer.Tokenizer(input_vocab_file_path, output_vocab_file_path)
    # Get test set
    test_input_ids, test_target_ids = get_test_dataset(test_file_path, tokenizer)
    # --------
    # print(test_input_ids[0], test_target_ids[0])
    # --------

    evaluate(test_input_ids, test_target_ids)
