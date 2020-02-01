import argparse
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import data
import tokenizer
import model

# Training settings
parser = argparse.ArgumentParser(description='Transformer for morphological inflection')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--train-file', type=str, default='data', metavar='S',
                    help="Train file of the dataset")
parser.add_argument('--valid-file', type=str, default='data', metavar='S',
                    help="Validation file of the dataset")
parser.add_argument('--vocab-file', type=str, default='data', metavar='S',
                    help="Base name of vocabulary files")
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--plots-folder', type=str, default='plots', metavar='D',
#                     help='how many batches to wait before logging training status')
# parser.add_argument('--model-folder', type=str, default='model', metavar='D',
#                     help='how many batches to wait before logging training status')
args = parser.parse_args()

# Set parameters for project
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.TransformerModel(src_vocab_size=51, tgt_vocab_size=44, embedding_dim=64, fcn_hidden_dim=256, num_heads=4,
                               num_layers=2, dropout=0.2)
model.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def get_train_dataset(train_file_path, tokenizer):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(train_file_path)
    # Pad target with sos and eos symbols
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Split target into to targets, for teacher forcing
    targets_tokens = [target_tokens[:-1] for target_tokens in outputs_tokens]
    targets_y_tokens = [target_tokens[1:] for target_tokens in outputs_tokens]

    # Get tensors of all input ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    # Get tensors of all target ids
    targets_ids = tokenizer.get_id_tensors(targets_tokens, device, "OUTPUT")
    # Get tensors of all result ids
    targets_y_ids = tokenizer.get_id_tensors(targets_y_tokens, device, "OUTPUT")
    return inputs_ids, targets_ids, targets_y_ids


def train(epoch, input_ids, target_ids, target_y_ids):
    """ Runs full training epoch over the dataset, uses teacher forcing in training
	CHECK EACH TO MAKE input_ids, target_ids GLOBAL"""
    model.train()
    running_loss = 0.0
    # Go over each example
    for i, (data, target, target_y) in enumerate(zip(input_ids, target_ids, target_y_ids)):
        optimizer.zero_grad()
        # Compute output of model
        output = model(data, target)
        # Compute loss over output
        loss = criterion(output, target_y)
        # Propagate loss and update model parameters
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print statistics
    print("Train Epoch: %d, loss: %.5f" % (epoch + 1, running_loss / (i + 1)))


def validation(epoch, input_ids, target_ids, target_y_ids):
    """ Runs full validation epoch over the dataset, uses teacher forcing in training
	CHECK EACH TO MAKE input_ids, target_ids GLOBAL"""
    model.eval()
    running_loss = 0.0
    # Go over each example
    for i, (data, target, target_y) in enumerate(zip(input_ids, target_ids, target_y_ids)):
        # Compute output of model
        output = model(data, target)
        # Compute loss over output
        loss = criterion(output, target_y)
        # Propagate loss and update model parameters
        running_loss += loss.item()
    # print statistics
    print("Validation. Epoch: %d, loss: %.5f" % (epoch + 1, running_loss / (i + 1)))


if __name__ == '__main__':
    # Get location of current folder, to work with full file paths
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    train_file_path = os.path.join(__location__, args.train_file)
    valid_file_path = os.path.join(__location__, args.valid_file)
    # Get vocabulary paths
    vocab_file_path = os.path.join(__location__, args.vocab_file)
    input_vocab_file_path = vocab_file_path + "-input"
    output_vocab_file_path = vocab_file_path + "-output"
    # Initialize tokenizer object with input and output vocabulary files
    tokenizer = tokenizer.Tokenizer(input_vocab_file_path, output_vocab_file_path)
    # Get training , validation and test sets
    train_input_ids, train_target_ids, train_target_y_ids = get_train_dataset(train_file_path, tokenizer)
    valid_input_ids, valid_target_ids, valid_target_y_ids = get_train_dataset(valid_file_path, tokenizer)
    # --------
    print(train_input_ids[0], train_target_ids[0], train_target_y_ids[0])
    # --------

    for epoch in range(1, args.epochs + 1):
        print("Train Epoch: %d started" % epoch)
        train(epoch, train_input_ids, train_target_ids, train_target_y_ids)
        validation(epoch, valid_input_ids, valid_target_ids, valid_target_y_ids)
        model_file = 'checkpoints/model_' + str(epoch) + '.pth'
        torch.save(model, model_file)
        print('\nSaved model to', model_file)
