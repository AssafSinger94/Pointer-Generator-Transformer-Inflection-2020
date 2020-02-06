from random import shuffle
import torch

import data
# import tokenizer


def get_train_dataset(train_file_path, tokenizer, max_seq_len=25):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(train_file_path)
    # Pad target with sos and eos symbols
    inputs_tokens = tokenizer.add_sequence_symbols(inputs_tokens)
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Split target into two targets, for teacher forcing
    targets_tokens = [target_tokens[:-1] for target_tokens in outputs_tokens]
    targets_y_tokens = [target_tokens[1:] for target_tokens in outputs_tokens]

    # Get lists of all input ids, target ids and target_y ids, where each sequence padded up to max length
    inputs_ids = [
        tokenizer.convert_input_tokens_to_ids(tokenizer.pad_tokens_sequence(input_tokens, max_seq_len))
        for input_tokens in inputs_tokens]
    targets_ids = [
        tokenizer.convert_output_tokens_to_ids(tokenizer.pad_tokens_sequence(target_tokens, max_seq_len))
        for target_tokens in targets_tokens]
    targets_y_ids = [
        tokenizer.convert_output_tokens_to_ids(tokenizer.pad_tokens_sequence(target_y_tokens, max_seq_len))
        for target_y_tokens in targets_y_tokens]

    return inputs_ids, targets_ids, targets_y_ids


def get_valid_dataset(valid_file_path, tokenizer, device, max_seq_len):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens, outputs_tokens = data.read_train_file_tokens(valid_file_path)
    # Pad target with sos and eos symbols
    outputs_tokens = tokenizer.add_sequence_symbols(outputs_tokens)
    # Pad with padding
    inputs_tokens = [tokenizer.pad_tokens_sequence(input_tokens, max_seq_len) for
                     input_tokens in inputs_tokens]
    outputs_tokens = [output_tokens for
                      output_tokens in outputs_tokens]

    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    outputs_ids = tokenizer.get_id_tensors(outputs_tokens, device, "OUTPUT")

    return inputs_ids, outputs_ids



def get_test_dataset(test_file_path, tokenizer):
    """ Reads entire dataset file, tokenizes it, and converts all tokens to ids using given tokenizer object """
    # Read dataset file, and get input tokens and output tokens from file
    inputs_tokens = data.read_test_file_tokens(test_file_path)
    # Get tensors of all input ids and output ids
    inputs_ids = tokenizer.get_id_tensors(inputs_tokens, device, "INPUT")
    return inputs_ids


def split_to_batches(ids_list, device, batch_size=128):
    """ splits list of id sequence into batchs.
        Gets list of sequences (list of size seq_len)
        returns list of batchs, each batch is a tensor of size N x S (batch_size x seq_len)"""
    return [torch.tensor(ids_list[x:x + batch_size], dtype=torch.long, device=device) for x in
            range(0, len(ids_list), batch_size)]


def get_batches(input_ids, target_ids, target_y_ids, device, batch_size=128):
    """ Gets entire dataset, shuffles the data, and split it to batchs.
        Each batch is a tensor of size N x S (batch_size x seq_len)."""
    # Get indexes of all dataset examples in shuffled order
    indexes = [i for i in range(len(input_ids))]
    shuffle(indexes)
    # get new dataset values, in shuffled order
    shuffled_input_ids = [input_ids[i] for i in indexes]
    shuffled_target_ids = [target_ids[i] for i in indexes]
    shuffled_target_y_ids = [target_y_ids[i] for i in indexes]
    # split to batches
    input_ids_batches = split_to_batches(shuffled_input_ids, device, batch_size)
    target_ids_batches = split_to_batches(shuffled_target_ids, device, batch_size)
    target_y_ids_batches = split_to_batches(shuffled_target_y_ids, device, batch_size)
    return input_ids_batches, target_ids_batches, target_y_ids_batches


class DataLoader(object):
    """ Contains all utilities for reading train/valid/test sets """
    def __init__(self, tokenizer, train_file_path, valid_file_path, device):
        self.tokenizer = tokenizer
        self.device = device
        # Read train file and get train set
        train_input_ids, train_target_ids, train_target_y_ids = get_train_dataset(train_file_path, tokenizer)
        self.train_input_ids = train_input_ids
        self.train_target_ids = train_target_ids
        self.train_target_y_ids = train_target_y_ids

        # Read validation file and get validation set
        valid_input_ids, valid_target_ids, valid_target_y_ids = get_train_dataset(valid_file_path, tokenizer)
        self.valid_input_ids = valid_input_ids
        self.valid_target_ids = valid_target_ids
        self.valid_target_y_ids = valid_target_y_ids

    def get_train_set(self):
        return get_batches(self.train_input_ids, self.train_target_ids, self.train_target_y_ids, self.device, batch_size=128)

    def get_validation_set(self):
        return get_batches(self.valid_input_ids, self.valid_target_ids, self.valid_target_y_ids, self.device, batch_size=128)

    def get_validation_set_len(self):
        return len(self.valid_input_ids)

    def get_padding_masks(self, data_batch, target_batch):
        """" Returns padding masks for entire dataset.
            Padding masks are ByteTensor where True values are positions that are masked and False values are not.
            inputs are of size N x S (batch_size x seq_len)
            Returns masks of same size- N x S (batch_size x seq_len) """
        # Get pad_id from tokenizer
        pad_token_id = self.tokenizer.pad_id
        # Get lists of all input ids, target ids and target_y ids, where each sequence padded up to max length
        src_padding_mask = (data_batch == pad_token_id)
        mem_padding_mask = src_padding_mask
        target_padding_mask = (target_batch == pad_token_id)
        return src_padding_mask, mem_padding_mask, target_padding_mask
