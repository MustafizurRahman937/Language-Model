import torch
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from importlib.util import (
    spec_from_file_location,
    module_from_spec
)


# --------------------------------------------------
# Locate files
# --------------------------------------------------

current_directory = os.path.dirname(
    os.path.abspath(__file__)
)

human_file = os.path.join(
    current_directory,
    "Human.csv"
)

ai_file = os.path.join(
    current_directory,
    "AI.csv"
)

model_file = os.path.join(
    current_directory,
    "2.py"
)

trained_model_file = os.path.join(
    current_directory,
    "simple_slm.pth"
)

results_file = os.path.join(
    current_directory,
    "prototype_results.txt"
)


# --------------------------------------------------
# Import SimpleSLM from 2.py
# --------------------------------------------------

spec = spec_from_file_location(
    "slm_model",
    model_file
)

if spec is None or spec.loader is None:

    raise ImportError(
        "Could not load the model from 2.py"
    )

slm_model = module_from_spec(spec)

spec.loader.exec_module(
    slm_model
)

SimpleSLM = slm_model.SimpleSLM


# --------------------------------------------------
# Load Human dataset
# --------------------------------------------------

human_data = pd.read_csv(
    human_file
)

print("Human dataset loaded.")

print(
    "Human samples:",
    len(human_data)
)


# --------------------------------------------------
# Load AI dataset
# --------------------------------------------------

ai_data = pd.read_csv(
    ai_file
)

print("\nAI dataset loaded.")

print(
    "AI samples:",
    len(ai_data)
)


# --------------------------------------------------
# Combine datasets
# --------------------------------------------------

data = pd.concat(
    [
        human_data,
        ai_data
    ],
    ignore_index=True
)

print("\nCombined dataset loaded.")

print(
    "Total samples:",
    len(data)
)


# --------------------------------------------------
# Remove empty text
# --------------------------------------------------

data["text"] = (
    data["text"]
    .astype(str)
    .str.strip()
)

data = data[
    data["text"] != ""
].reset_index(drop=True)


# --------------------------------------------------
# Prepare texts and labels
# --------------------------------------------------

texts = (
    data["text"]
    .astype(str)
    .tolist()
)

labels = (
    data["label"]
    .astype(int)
    .tolist()
)


# --------------------------------------------------
# 80/20 TEST SPLIT
# --------------------------------------------------

(
    train_texts,
    test_texts,
    train_labels,
    test_labels
) = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


print("\nTest dataset prepared.")

print(
    "Test samples:",
    len(test_texts)
)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

print(
    "Tokenizer loaded successfully."
)


# --------------------------------------------------
# Tokenize test data
# --------------------------------------------------

print("\nTokenizing test data...")

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=64
)

print(
    "Tokenization completed."
)


# --------------------------------------------------
# Create PyTorch Dataset
# --------------------------------------------------

class TextDataset(Dataset):

    def __init__(
        self,
        encodings,
        labels
    ):

        self.encodings = encodings
        self.labels = labels


    def __len__(self):

        return len(
            self.labels
        )


    def __getitem__(
        self,
        index
    ):

        item = {
            key: torch.tensor(
                value[index],
                dtype=torch.long
            )
            for key, value
            in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


# --------------------------------------------------
# Create test dataset
# --------------------------------------------------

test_dataset = TextDataset(
    test_encodings,
    test_labels
)


# --------------------------------------------------
# Create test DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# --------------------------------------------------
# Initialize model
# --------------------------------------------------

model = SimpleSLM(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    num_classes=2,
    max_length=64
)


# --------------------------------------------------
# Select device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        trained_model_file,
        map_location=device
    )
)

model.to(device)

model.eval()


print(
    "\nTrained ALBERT-inspired prototype loaded."
)

print(
    "Device:",
    device
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\n==============================================")
print("             TEST PREDICTIONS")
print("==============================================")


correct = 0

total = 0

test_results = []


# --------------------------------------------------
# Run predictions
# --------------------------------------------------

with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels_batch = batch[
            "labels"
        ].to(device)


        # --------------------------------------------------
        # Model prediction
        # --------------------------------------------------

        outputs = model(
            input_ids,
            attention_mask
        )


        # --------------------------------------------------
        # Convert outputs to probabilities
        # --------------------------------------------------

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # --------------------------------------------------
        # Get predictions
        # --------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # --------------------------------------------------
        # Process each test sample
        # --------------------------------------------------

        for i in range(
            len(labels_batch)
        ):

            actual_label = (
                labels_batch[i].item()
            )

            predicted_label = (
                predictions[i].item()
            )

            ai_probability = (
                probabilities[i][1].item()
            )


            # --------------------------------------------------
            # Convert labels to names
            # --------------------------------------------------

            if actual_label == 0:

                actual_text = "Human"

            else:

                actual_text = "AI"


            if predicted_label == 0:

                predicted_text = "Human"

            else:

                predicted_text = "AI"


            # --------------------------------------------------
            # Calculate correctness
            # --------------------------------------------------

            if (
                actual_label
                ==
                predicted_label
            ):

                correct += 1


            total += 1


            # --------------------------------------------------
            # Save result
            # --------------------------------------------------

            test_results.append({
                "sample": total,
                "actual": actual_text,
                "predicted": predicted_text,
                "ai_probability": ai_probability
            })


# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------

accuracy = (
    correct
    /
    total
)


# --------------------------------------------------
# Display individual predictions
# --------------------------------------------------

for result in test_results:

    print(
        f"\nTest Sample "
        f"{result['sample']}"
    )

    print(
        "Actual:",
        result["actual"]
    )

    print(
        "Predicted:",
        result["predicted"]
    )

    print(
        "AI Probability:",
        f"{result['ai_probability'] * 100:.2f}%"
    )


# --------------------------------------------------
# FINAL EVALUATION SUMMARY
# --------------------------------------------------

print("\n==============================================")

print(
    "       ALBERT-INSPIRED PROTOTYPE EVALUATION"
)

print("==============================================")


print("\nCorrect Predictions:")

print(
    correct
)


print("\nTotal Test Samples:")

print(
    total
)


print("\nPrototype Accuracy:")

print(
    f"{accuracy * 100:.2f}%"
)


print("\n==============================================")

print(
    "          EVALUATION COMPLETED"
)

print("==============================================")


# --------------------------------------------------
# Save results to text file
# --------------------------------------------------

with open(
    results_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "==============================================\n"
    )

    file.write(
        "       ALBERT-INSPIRED PROTOTYPE EVALUATION\n"
    )

    file.write(
        "==============================================\n\n"
    )


    file.write(
        "TEST PREDICTIONS\n"
    )

    file.write(
        "----------------------------------------------\n"
    )


    for result in test_results:

        file.write(
            f"\nTest Sample {result['sample']}\n"
        )

        file.write(
            f"Actual: {result['actual']}\n"
        )

        file.write(
            f"Predicted: {result['predicted']}\n"
        )

        file.write(
            f"AI Probability: "
            f"{result['ai_probability'] * 100:.2f}%\n"
        )


    file.write(
        "\n\n==============================================\n"
    )

    file.write(
        "       ALBERT-INSPIRED PROTOTYPE EVALUATION\n"
    )

    file.write(
        "==============================================\n\n"
    )

    file.write(
        f"Correct Predictions: {correct}\n"
    )

    file.write(
        f"Total Test Samples: {total}\n"
    )

    file.write(
        f"Prototype Accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )


# --------------------------------------------------
# Completion message
# --------------------------------------------------

print(
    "\nEvaluation results saved successfully."
)

print(
    "Saved at:",
    results_file
)

