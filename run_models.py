import os

LANGUAGES = ["english", "french", "hebrew", "irish", "italian", "spanish"]
RESOURCES = ["low"]#"medium"]
MODEL_TYPE = ["transformer", "pointer_generator"]
EPOCHS_PER_RESOURCE = {"low" : 400, "medium" : 200}
BATCH_SIZE_PER_RESOURCE = {"low" : 64, "medium" : 128}
for model in MODEL_TYPE:
    for resource in RESOURCES:
        for language in LANGUAGES:
            print("{} - {}".format(resource, language))
            # Get epoch and batch size
            epochs = EPOCHS_PER_RESOURCE[resource]
            # -------
            # epochs = 3
            # -------
            batch_size = BATCH_SIZE_PER_RESOURCE[resource]
            # Set names of relevant files and directories
            train_file_name = "{}-{}-{}".format(language, "train", resource)
            valid_file_name = "{}-{}".format(language, "dev")
            test_file_name = "{}-{}".format(language, "test")
            covered_test_file_name = "{}-{}-{}".format(language, "covered", "test")
            prediction_file_name = "{}-{}-{}-{}".format(language, resource, "test", "pred")
            vocab_file_name = "{}-{}".format(train_file_name, "vocab")

            vocab_folder = "vocab/{}/{}".format(language, resource)
            checkpoints_folder = "model-checkpoints/{}/{}/{}".format(model, language, resource)
            predictions_folder = "predictions/{}".format(model)
            logs_folder = "logs"

            # create vocabulary, checkpoints and predictions folders, if they do not exist already
            if not os.path.exists(vocab_folder):
                os.makedirs(vocab_folder)
            if not os.path.exists(checkpoints_folder):
                os.makedirs(checkpoints_folder)
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)
            if not os.path.exists(logs_folder):
                os.makedirs(logs_folder)

            # Create vocabulary
            os.system("python vocabulary.py --src-file {} ".format(train_file_name) +
                      "--vocab-file {}/{}".format(vocab_folder, vocab_file_name))
            # Train model
            os.system("python main.py --model {} --epochs {} --batch-size {} ".format(model, epochs, batch_size) +
                      "--train-file {} --valid-file {} ".format(train_file_name, valid_file_name) +
                      "--vocab-file {}/{} ".format(vocab_folder, vocab_file_name) +
                      "--checkpoints-folder {}".format(checkpoints_folder) +
                      " > {}/train-log-{}-{}-{}.out".format(logs_folder, model, resource, language))
            # Generate predictions for test set
            os.system("python generate.py --model-checkpoint {}/model_best.pth ".format(checkpoints_folder) +
                      "--valid-file {} --test-file {} ".format(valid_file_name, covered_test_file_name) +
                      "--vocab-file {}/{} ".format(vocab_folder, vocab_file_name) +
                      "--out-file {}/{}".format(predictions_folder, prediction_file_name))
            # Evaluate accuracy of prediction file compared to true test set
            os.system("python evaluate.py --pred-file {}/{} ".format(predictions_folder, prediction_file_name) +
                      "--target-file {}".format(test_file_name))
