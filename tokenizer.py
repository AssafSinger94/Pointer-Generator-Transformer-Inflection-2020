import collections

import torch


def load_vocab(vocab_file_path, sos, eos, pad):
    """ Loads vocabulary from vocabulary file (create by vocabulary.py)"""
    vocab_file = open(vocab_file_path, "r+", encoding='utf-8')
    lines = vocab_file.readlines()

    vocab = collections.OrderedDict()
    # First, add special signs for sos, eos and pad to vocabulary
    vocab[sos] = 0
    vocab[eos] = 1
    vocab[pad] = 2
    # For each valid line, Get token and index of line
    for index, line in enumerate(lines):
        if line != "\n":
            token, count = line.replace("\n", "").split("\t")
            vocab[token] = index + 3  # first two values of vocabulary are taken

    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of tokens or ids using the given vocabulary."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


def tokenize_words(word_list):
    """Split words to lists of characters"""
    return [list(words) for words in word_list]


def tokenize_features(features_list):
    """Splits features by the separator sign ";" """
    return [connected_features.split(";") for connected_features in features_list]


class Tokenizer(object):
    """ Tokenizer object. Handles tokenizing sentences, converting tokens to ids and vice versa"""

    def __init__(self, input_vocab_file_path, output_vocab_file_path):
        self.sos = "<s>"
        self.eos = "</s>"
        self.pad = "<pad>"
        self.sos_id = 0
        self.eos_id = 1
        self.pad_id = 2

        self.input_vocab = load_vocab(input_vocab_file_path, self.sos,
                                      self.eos, self.pad)  # vocabulary of all token->id in the input
        self.inv_input_vocab = {v: k for k, v in self.input_vocab.items()}  # reverse vocabulary of input, id->token
        self.output_vocab = load_vocab(output_vocab_file_path, self.sos,
                                       self.eos, self.pad)  # vocabulary of all token->id in the output
        self.inv_output_vocab = {v: k for k, v in self.output_vocab.items()}  # reverse vocabulary of output, id->token

    def add_sequence_symbols(self, tokens_list):
        """ Adds eos and sos symbols to each sequence of tokens"""
        return [[self.sos] + tokens + [self.eos] for tokens in tokens_list]

    def convert_input_tokens_to_ids(self, tokens):
        """ Converts all given tokens to ids using the input vocabulary"""
        return convert_by_vocab(self.input_vocab, tokens)

    def convert_input_ids_to_tokens(self, ids):
        """ Converts all given ids to tokens using the input vocabulary"""
        return convert_by_vocab(self.inv_input_vocab, ids)

    def get_input_vocab_size(self):
        """ Returns size of input vocabulary """
        return len(self.input_vocab)

    def convert_output_tokens_to_ids(self, tokens):
        """ Converts all given tokens to the ids using the output vocabulary"""
        return convert_by_vocab(self.output_vocab, tokens)

    def convert_output_ids_to_tokens(self, ids):
        """ Converts all given tokens to the ids using the output vocabulary"""
        return convert_by_vocab(self.inv_output_vocab, ids)

    def get_output_vocab_size(self):
        """ Returns size of output vocabulary """
        return len(self.output_vocab)

    def get_id_tensors(self, tokens_list, device, vocab_type):
        """ Gets list of token sequences, and converts each token sequence to tensor of ids, using the tokenizer
            device to determine tensor device type, and vocab type is either "INPUT" or "OUTPUT" """
        if vocab_type == "INPUT":
            return [torch.tensor(self.convert_input_tokens_to_ids(tokens), dtype=torch.long, device=device)
                    for tokens in tokens_list]
        else:
            return [torch.tensor(self.convert_output_tokens_to_ids(tokens), dtype=torch.long, device=device)
                    for tokens in tokens_list]

    def pad_tokens_sequence(self, tokens, max_seq_len):
        """ Pads the token sequence with pad symbols until it reaches the max sequence length.
            If Sequence is already at max length, nothing is added. """
        padding_len = max_seq_len - len(tokens)
        padding = [self.pad] * padding_len
        return tokens + padding
