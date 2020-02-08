# Transformer-Morph
Transformer model for the morphological inflection task

Train model - python main.py --train-file "english-train-medium" --valid-file "english-dev" --vocab-file "english-train-medium-vocab"

Test model - python generate.py --model "checkpoints/model_best.pth" --valid-file "english-dev" --test-file "english-dev-as-test" --vocab-file "english-train-medium-vocab" --out-file "english-pred"

Compute accuracy of test set predictions - python evaluate.py --pred-file "english-pred" --target-file "english-dev"