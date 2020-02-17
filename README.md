# Transformer-Morph
Transformer model for the morphological inflection task

---MEDIUM RESOURCE TRAINING FILE---
Train model - python main.py --epochs 150 --batch-size 128 --train-file "english-train-medium" --valid-file "english-dev" --vocab-file "english-train-medium-vocab"

Test model - python generate.py --model "checkpoints/model_best.pth" --valid-file "english-dev" --test-file "english-covered-test" --vocab-file "english-train-medium-vocab" --out-file "english-test-pred-medium"

Compute accuracy of test set predictions - python evaluate.py --pred-file "english-test-pred-medium" --target-file "english-test"


---LOW RESOURCE TRAINING FILE---
Train model - python main.py --epochs 400 --batch-size 64 --train-file "english-train-low" --valid-file "english-dev" --vocab-file "english-train-low-vocab"

Test model - python generate.py --model "checkpoints/model_best.pth" --valid-file "english-dev" --test-file "english-covered-test" --vocab-file "english-train-low-vocab" --out-file "english-test-pred-low"

Compute accuracy of test set predictions - python evaluate.py --pred-file "english-test-pred-low" --target-file "english-test"