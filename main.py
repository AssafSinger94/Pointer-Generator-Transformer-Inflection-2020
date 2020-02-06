import argparse
import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataset
import tokenizer
import model
from data import DATA_FOLDER

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

# Get train and validation file paths
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
train_file_path = os.path.join(__location__, DATA_FOLDER, args.train_file)
valid_file_path = os.path.join(__location__, DATA_FOLDER, args.valid_file)
# Get vocabulary paths
input_vocab_file_path = os.path.join(__location__, DATA_FOLDER, args.vocab_file + "-input")
output_vocab_file_path = os.path.join(__location__, DATA_FOLDER, args.vocab_file + "-output")
# Initialize Tokenizer object with input and output vocabulary files
myTokenizer = tokenizer.Tokenizer(input_vocab_file_path, output_vocab_file_path)

""" CONSTANTS """
MAX_INPUT_SEQ_LEN = 25
MAX_OUTPUT_SEQ_LEN = 25
SRC_VOCAB_SIZE = myTokenizer.get_input_vocab_size()
TGT_VOCAB_SIZE = myTokenizer.get_output_vocab_size()
# Model Hyperparameters
EMBEDDING_DIM = 64
FCN_HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2


""" MODEL AND DATA LOADER """
# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.TransformerModel(src_vocab_size=SRC_VOCAB_SIZE, tgt_vocab_size=TGT_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                               fcn_hidden_dim=FCN_HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT)
model.to(device)
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=myTokenizer.pad_id)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Initialize DataLoader object
data_loader = dataset.DataLoader(myTokenizer, train_file_path, valid_file_path, device)


def train(epoch):
    """ Runs full training epoch over the training set, uses teacher forcing in training"""
    model.train()
    running_loss = 0.0
    # Get Training set in batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_train_set()
    # Go over each batch
    for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        optimizer.zero_grad()
        # Get padding masks
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        # Compute output of model
        output = model(data, target, src_key_padding_mask=src_pad_mask, memory_key_padding_mask=mem_pad_mask,
                       tgt_key_padding_mask=target_pad_mask)
        # Compute loss
        loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), target_y.contiguous().view(-1))
        # Propagate loss and update model parameters
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # print statistics
    print("Train Epoch: %d, loss: %.5f" % (epoch, running_loss / (i + 1)))


def validation(epoch):
    """ Computes loss and accuracy over the validation set, using teacher forcing inputs """
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    # Get Training set in batches
    input_ids_batches, target_ids_batches, target_y_ids_batches = data_loader.get_validation_set()
    # Go over each batch
    for i, (data, target, target_y) in enumerate(zip(input_ids_batches, target_ids_batches, target_y_ids_batches)):
        # Get padding masks
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target)
        # Compute output of model
        output = model(data, target, src_key_padding_mask=src_pad_mask, memory_key_padding_mask=mem_pad_mask,
                       tgt_key_padding_mask=target_pad_mask)
        # ----------
        # Get predictions
        predictions = F.softmax(output, dim=-1).topk(1)[1].squeeze()
        # Compute accuracy
        for j in range(target_y.shape[0]):
            pad_start_idx = target_pad_mask[j].nonzero()[0]
            if torch.all(torch.eq(predictions[j, :pad_start_idx], target_y[j, :pad_start_idx])):
                correct_preds += 1
        # ----------
        # Compute loss over output
        loss = criterion(output.contiguous().view(-1, TGT_VOCAB_SIZE), target_y.contiguous().view(-1))
        running_loss += loss.item()
    # print statistics
    final_loss = running_loss / (i + 1)
    accuracy = correct_preds / data_loader.get_validation_set_len()
    print("Validation. Epoch: %d, loss: %.4f, accuracy: %.4f" % (epoch, final_loss, accuracy))
    return final_loss


if __name__ == '__main__':
    # Initialize best validation loss placeholders
    best_valid_loss = sys.maxsize
    best_valid_epoch = 0
    best_model_file = 'checkpoints/model_best.pth'
    for epoch in range(1, args.epochs + 1):
        print("\nTrain Epoch: %d started" % epoch)
        # Train model
        train(epoch)
        # Check model on validation set and get loss
        curr_valid_loss = validation(epoch)
        # Save model with epoch number
        model_file = "checkpoints/model_%d.pth" % epoch
        torch.save(model, model_file)
        print('Saved model to', model_file)
        # If best model so far, save model as best and the loss values
        if curr_valid_loss <= best_valid_loss:
            best_valid_loss = curr_valid_loss
            best_valid_epoch = epoch
            shutil.copyfile(model_file, best_model_file)
            print("New best Loss, saved to %s" % best_model_file)

    print('\nFinished training, best model on validation set: ', best_valid_epoch)
