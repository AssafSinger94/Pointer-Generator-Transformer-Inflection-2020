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
parser.add_argument('--valid-file', type=str, default='data', metavar='S',
                    help="Test file of the dataset")
parser.add_argument('--vocab-file', type=str, default='data', metavar='S',
                    help="Base name of vocabulary files")
parser.add_argument('--model', type=str, default='checkpoints/model_best.pth', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='gtsrb_kaggle.csv', metavar='D',
                    help="name of the output csv file")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model from checkpoint
model = torch.load(args.model)
model.eval()

MAX_SEQ_LEN = 25
PAD_ID = 2


def get_test_dataset(test_file_path, tokenizer):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens = data.read_test_file_tokens(test_file_path)
    # Get tensors of all input ids and output ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    return inputs_ids


def get_valid_dataset(valid_file_path, tokenizer):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(valid_file_path)
    # Pad target with sos and eos symbols
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Pad with padding
    inputs_tokens = [tokenizer.pad_tokens_sequence(input_tokens, MAX_SEQ_LEN) for
                     input_tokens in inputs_tokens]
    outputs_tokens = [output_tokens for
                      output_tokens in outputs_tokens]

    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    outputs_ids = tokenizer.get_id_tensors(outputs_tokens, device, "OUTPUT")

    return inputs_ids, outputs_ids


def get_padding_masks(data_batch, target_batch):
    """" Returns padding masks for entire dataset.
        inputs are of size batch_size x seq_len
        Returns masks of same size- batch_size x seq_len """
    pad_token_id = PAD_ID
    # Get lists of all input ids, target ids and target_y ids, where each sequence padded up to max length
    src_padding_mask = (data_batch == pad_token_id)
    mem_padding_mask = src_padding_mask
    target_padding_mask = (target_batch == pad_token_id)
    return src_padding_mask, mem_padding_mask, target_padding_mask


def evaluate_validation(input_ids, target_ids, max_seq_len=MAX_SEQ_LEN):
    """ Runs full validation epoch over the dataset
	CHECK EACH TO MAKE input_ids, target_ids GLOBAL"""
    correct = 0
    # Go over each example
    for i, (data, target) in enumerate(zip(input_ids, target_ids)):
        # Add batch dimension
        data = data.unsqueeze(dim=0)
        src_pad_mask, mem_pad_mask, target_pad_mask = get_padding_masks(data, target.unsqueeze(dim=0))
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
    # Get location of current folder, to work with full file paths
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    valid_file_path = os.path.join(__location__, args.valid_file)
    # Get vocabulary paths
    vocab_file_path = os.path.join(__location__, args.vocab_file)
    input_vocab_file_path = vocab_file_path + "-input"
    output_vocab_file_path = vocab_file_path + "-output"
    # Initialize tokenizer object with input and output vocabulary files
    tokenizer = tokenizer.Tokenizer(input_vocab_file_path, output_vocab_file_path)
    # Get test set
    valid_input_ids, valid_target_ids = get_valid_dataset(valid_file_path, tokenizer)
    # --------
    # print(test_input_ids[0], test_target_ids[0])
    # --------

    evaluate_validation(valid_input_ids, valid_target_ids)

# def get_valid_dataset(valid_file_path, tokenizer):
#     """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
#     # Read dataset file, and get input tokens and output tokens from file
#     inputs_tokens, outputs_tokens = data.read_train_file_tokens(valid_file_path)
#     # Pad target with sos and eos symbols
#     outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
#     # Get tensors of all input ids and output ids
#     inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
#     output_ids = tokenizer.get_id_tensors(outputs_tokens, device, "OUTPUT")
#     return inputs_ids, output_ids
