# Transformer-Morph
Transformer model for the morphological inflection task

**MEDIUM RESOURCE TRAINING FILE**

Create vocabulary for dataset - python vocabulary.py --src-file "english-train-medium" --vocab-file "data/english-train-medium-vocab"

Train model - python main.py --epochs 150 --batch-size 128 --train-file "english-train-medium" --valid-file "english-dev" --vocab-file "data/english-train-medium-vocab" checkpoints-folder "checkpoints"

Test model - python generate.py --model-checkpoint "checkpoints/model_best.pth" --valid-file "english-dev" --test-file "english-covered-test" --vocab-file "data/english-train-medium-vocab" --out-file "data/english-test-pred-medium"

Compute accuracy of test set predictions - python evaluate.py --pred-file "data/english-test-pred-medium" --target-file "english-test"


**LOW RESOURCE TRAINING FILE**

Create vocabulary for dataset - python vocabulary.py --src-file "english-train-low" --vocab-file "data/english-train-low-vocab"

Train model - python main.py --epochs 400 --batch-size 64 --train-file "english-train-low" --valid-file "english-dev" --vocab-file "data/english-train-low-vocab"  checkpoints-folder "checkpoints"

Test model - python generate.py --model-checkpoint "checkpoints/model_best.pth" --valid-file "english-dev" --test-file "english-covered-test" --vocab-file "data/english-train-low-vocab" --out-file "data/english-test-pred-low"

Compute accuracy of test set predictions - python evaluate.py --pred-file "data/english-test-pred-low" --target-file "english-test"