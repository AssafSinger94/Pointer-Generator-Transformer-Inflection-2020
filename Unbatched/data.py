import os
import re
import tokenizer

"""
Allows reading of conll files.
Reads conll file, and 1) Splits it by different components of each examples, and also 2) Seperates it to to input tokens and target tokens.
conll: lemma-tab-target-tab-features
"""

""" Reads conll file, split to line, and splits each line by tabs
	Returns list of lists"""
def read_morph_file(morph_file_path):
	# Read lines of file
	morph_file = open(morph_file_path, 'r', encoding='utf-8')
	lines = morph_file.readlines()

	outputs = []
	# Seperate lines to proper format
	for line in lines:
		if line != "\n":
			# Strip '\n' and split
			outputs.append(line.replace("\n", "").split("\t"))
	morph_file.close()
	return outputs

""" Reads conll train file, and splits to lists of input lemmas, input features and target lemmas"""


def clean_word(word):
	word = re.sub("[!@#$']", '', word)
	return word.lower()


def read_train_file(train_file_path):
	lemmas = []
	targets = []
	features = []
	train_morph_list = read_morph_file(train_file_path)

	for lemma, target, feature in train_morph_list:
		# Add results to relevant lists
		lemmas.append(clean_word(lemma))
		features.append(feature)
		targets.append(clean_word(target))

	return lemmas, targets, features


""" Reads conll test file, and splits to lists of input lemmas, input features and target lemmas"""
def read_test_file(test_input_file):
	lemmas = []
	features = []
	test_morph_list = read_morph_file(test_input_file)

	for lemma, feature in test_morph_list:
		# Add results to relevant lists
		lemmas.append(lemma)
		features.append(feature)

	return lemmas, features


""" Reads conll train file, and splits to input tokens and target tokens.
	Each input and target is a list of tokens"""
def read_train_file_tokens(train_file_path):
	lemmas, targets, features = read_train_file(train_file_path)
	# tokenize all three lists, get as list of tokens lists
	lemmas_tokens = tokenizer.tokenize_words(lemmas)
	targets_tokens = tokenizer.tokenize_words(targets)
	features_tokens = tokenizer.tokenize_features(features)
	# concatenate feature tokens to lemma tokens
	input_tokens = [lemma_tokens + feature_tokens for lemma_tokens, feature_tokens in zip(lemmas_tokens, features_tokens)]
	return input_tokens, targets_tokens