# Transformer-Morph
Transformer model for the morphological inflection task

Train model - python main.py --train-file "english-train-medium" --valid-file "english-dev" --vocab-file "english-train-medium-vocab"
Test model - python evaluate.py --model "checkpoints/model_best.pth" --valid-file "english-dev" --vocab-file "english-train-medium-vocab"
* Right now testing the model only computes accuracy over the validation set.
