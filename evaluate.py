import argparse
import os
import torch
# import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

import dataset
from data import DATA_FOLDER
import tokenizer
import model

# Training settings
parser = argparse.ArgumentParser(description='Transformer for morphological inflection')
parser.add_argument('--valid-file', type=str, default='data', metavar='S',
                    help="Test file of the dataset")
parser.add_argument('--vocab-file', type=str, default='data', metavar='S',
                    help="Base name of vocabulary files")
parser.add_argument('--model', type=str, default='checkpoints/model_best.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")
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
# MAX_INPUT_SEQ_LEN = 25
# MAX_OUTPUT_SEQ_LEN = 25
# SRC_VOCAB_SIZE = myTokenizer.get_input_vocab_size()
# TGT_VOCAB_SIZE = myTokenizer.get_output_vocab_size()
MAX_SEQ_LEN = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model from checkpoint
model = torch.load(args.model)
model.eval()

# Initialize DataLoader object
data_loader = dataset.DataLoader(myTokenizer, train_file_path, valid_file_path, device)


def evaluate_validation(max_seq_len=MAX_SEQ_LEN):
    """ Runs full validation epoch over the dataset
	CHECK EACH TO MAKE input_ids, target_ids GLOBAL"""
    input_ids, target_ids = data_loader.get_validation_set()

    correct = 0

    # Go over each example
    for i, (data, target) in enumerate(zip(input_ids, target_ids)):
        # Add batch dimension
        data = data.unsqueeze(dim=0)
        src_pad_mask, mem_pad_mask, target_pad_mask = data_loader.get_padding_masks(data, target.unsqueeze(dim=0))
        outputs = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
        outputs[0] = tokenizer.sos_id
        for j in range(1, max_seq_len):
            # Compute output of model
            out = model(data, outputs[:, :j],
                        src_key_padding_mask=src_pad_mask, memory_key_padding_mask=mem_pad_mask).squeeze()
            out = F.softmax(out, dim=-1)
            val, ix = out.topk(1)

            outputs[0, j] = ix[-1]
            if ix[-1] == tokenizer.eos_id:
                break
        pred = outputs[0, :j + 1]
        if torch.equal(pred, target) and torch.all(torch.eq(pred, target)):
            correct += 1
    print("Validation set: accuracy: %.3f" % (100 * correct / len(input_ids)))
    # return ' '.join(tokenizer.convert_output_ids_to_tokens(outputs[:i]))


#     def translate(model, src, max_len=80, custom_string=False):
#
#         model.eval()
#
#     if custom_sentence == True:
#         src = tokenize_en(src)
#         sentence = \
#             Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok
#                                         in sentence]])).cuda()
#     src_mask = (src != input_pad).unsqueeze(-2)
#     e_outputs = model.encoder(src, src_mask)
#
#     outputs = torch.zeros(max_len).type_as(src.data)
#     outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])
#
#
# for i in range(1, max_len):
#
#     trg_mask = np.triu(np.ones((1, i, i),
#                                k=1).astype('uint8')
#     trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
#
#     out = model.out(model.decoder(outputs[:i].unsqueeze(0),
#                                   e_outputs, src_mask, trg_mask))
#     out = F.softmax(out, dim=-1)
#     val, ix = out[:, -1].data.topk(1)
#
#     outputs[i] = ix[0][0]
#     if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
#         break


if __name__ == '__main__':
    # Get test set
    # valid_input_ids, valid_target_ids = get_valid_dataset(valid_file_path, tokenizer)
    # --------
    # print(test_input_ids[0], test_target_ids[0])
    # --------

    evaluate_validation()
