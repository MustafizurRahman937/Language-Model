import torch
import pandas as pd
import os
import time

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.model_selection import train_test_split


# ============================================================
# FILE PATHS


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

result_file = os.path.join(
    current_directory,
    "deberta_results.txt"
)

computational_file = os.path.join(
    current_directory,
    "deberta_computational_cost.txt"
)


# ============================================================
# LOAD DATASETS from file
# ============================================================

human_data = pd.read_csv(
    human_file
)

ai_data = pd.read_csv(
    ai_file
)

print("Human dataset loaded successfully.")

print(
    "Human samples:",
    len(human_data)
)

print("\nAI dataset loaded successfully.")

print(
    "AI samples:",
    len(ai_data)
)


# ============================================================
# COMBINE DATASETS
# ============================================================

data = pd.concat(
    [
        human_data,
        ai_data
    ],
    ignore_index=True
)

texts = data["text"].astype(str).tolist()

labels = data["label"].astype(int).tolist()

print("\nCombined dataset created.")

print(
    "Total samples:",
    len(data)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_texts, test_texts, train_labels, test_labels = (
    train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
)

print("\nDataset split completed.")

print(
    "Training samples:",
    len(train_texts)
)

print(
    "Testing samples:",
    len(test_texts)
)


# ============================================================
# LOAD DEBERTA TOKENIZER
# ============================================================

print("\nLoading DeBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-base"
)

print(
    "DeBERTa tokenizer loaded successfully."
)


# ============================================================
# TOKENIZATION
# ============================================================

print("\nTokenizing training data...")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=64
)

print(
    "Training text tokenization completed."
)

print("\nTokenizing test data...")

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=64
)

print(
    "Test text tokenization completed."
)


# ============================================================
# SHOW TOKENIZATION EXAMPLE
# ============================================================

print("\n==============================================")
print("             TOKENIZATION EXAMPLE")
print("==============================================")

print("\nOriginal text:")

print(train_texts[0])

print("\nTokens:")

print(
    tokenizer.convert_ids_to_tokens(
        train_encodings["input_ids"][0]
    )
)

print("\nToken IDs:")

print(
    train_encodings["input_ids"][0]
)


# ============================================================
# DATASET CLASS
# ============================================================

class TextDataset(Dataset):

    def __init__(
        self,
        encodings,
        labels
    ):

        self.encodings = encodings
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(
        self,
        index
    ):

        item = {
            key: torch.tensor(
                value[index],
                dtype=torch.long
            )
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = TextDataset(
    train_encodings,
    train_labels
)

test_dataset = TextDataset(
    test_encodings,
    test_labels
)


# ============================================================
# CREATE DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)


# ============================================================
# LOAD DEBERTA MODEL
# ============================================================

print("\n==============================================")
print("             LOADING DEBERTA MODEL")
print("==============================================")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-base",
    num_labels=2
)

print(
    "\nDeBERTa model loaded successfully."
)


# ============================================================
# COUNT PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Number of parameters:",
    total_parameters
)


# ============================================================
# SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

print(
    "Device:",
    device
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)


# ============================================================
# TRAINING
# ============================================================

epochs = 1

print("\n==============================================")
print("             DEBERTA TRAINING")
print("==============================================")


# ============================================================
# START TRAINING TIMER
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

training_start_time = time.perf_counter()


for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for batch in train_loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels_batch = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss

        logits = outputs.logits

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)


    average_loss = (
        total_loss /
        len(train_loader)
    )

    training_accuracy = (
        correct /
        total
    )

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: "
        f"{training_accuracy * 100:.2f}%"
    )


# ============================================================
# END TRAINING TIMER
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

training_end_time = time.perf_counter()

training_time = (
    training_end_time -
    training_start_time
)

print(
    f"\nTotal Training Time: "
    f"{training_time:.6f} seconds"
)


# ============================================================
# DEBERTA EVALUATION
# ============================================================

model.eval()

correct = 0

total = 0

results = []

print("\n==============================================")
print("            DEBERTA EVALUATION")
print("==============================================")


# ============================================================
# START INFERENCE TIMER
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

inference_start_time = time.perf_counter()


with torch.no_grad():

    for batch in test_loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels_batch = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

        for i in range(
            len(predictions)
        ):

            actual_label = (
                labels_batch[i].item()
            )

            predicted_label = (
                predictions[i].item()
            )

            ai_probability = (
                probabilities[i][1].item()
                * 100
            )

            if actual_label == 0:
                actual_text = "Human"
            else:
                actual_text = "AI"

            if predicted_label == 0:
                predicted_text = "Human"
            else:
                predicted_text = "AI"

            results.append(
                {
                    "text": test_texts[len(results)],
                    "actual": actual_text,
                    "predicted": predicted_text,
                    "ai_probability": ai_probability
                }
            )


# ============================================================
# END INFERENCE TIMER
# ============================================================

if device.type == "cuda":
    torch.cuda.synchronize()

inference_end_time = time.perf_counter()

inference_time = (
    inference_end_time -
    inference_start_time
)

average_inference_time = (
    inference_time /
    total
)


# ============================================================
# CALCULATE ACCURACY
# ============================================================

accuracy = correct / total

accuracy_percentage = accuracy * 100


# ============================================================
# DISPLAY TEST PREDICTIONS
# ============================================================

print("\n==============================================")
print("             TEST PREDICTIONS")
print("==============================================")


for number, result in enumerate(
    results,
    start=1
):

    print(
        f"\nTest Sample {number}"
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
        f"{result['ai_probability']:.2f}%"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("              DeBERTa Result")
print("==============================================")

print("\nCorrect Predictions:")

print(correct)

print("\nTotal Test Samples:")

print(total)

print("\nDeBERTa Accuracy:")

print(
    f"{accuracy_percentage:.2f}%"
)

print(
    f"\nTotal Inference Time: "
    f"{inference_time:.6f} seconds"
)

print(
    f"Average Inference Time per Sample: "
    f"{average_inference_time:.6f} seconds"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    result_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DeBERTa Evaluation\n"
    )

    file.write(
        "==================\n\n"
    )

    file.write(
        f"Total Test Samples: {total}\n"
    )

    file.write(
        f"Correct Predictions: {correct}\n"
    )

    file.write(
        f"DeBERTa Accuracy: "
        f"{accuracy_percentage:.2f}%\n\n"
    )

    file.write(
        "Test Predictions\n"
    )

    file.write(
        "----------------\n"
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        file.write(
            f"\nTest Sample {number}\n"
        )

        file.write(
            f"Actual: "
            f"{result['actual']}\n"
        )

        file.write(
            f"Predicted: "
            f"{result['predicted']}\n"
        )

        file.write(
            f"AI Probability: "
            f"{result['ai_probability']:.2f}%\n"
        )


# ============================================================
# SAVE COMPUTATIONAL COST RESULTS
# ============================================================

with open(
    computational_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DeBERTa Computational Cost\n"
    )

    file.write(
        "==========================\n\n"
    )

    file.write(
        f"Training Time: "
        f"{training_time:.6f} seconds\n"
    )

    file.write(
        f"Inference Time: "
        f"{inference_time:.6f} seconds\n"
    )

    file.write(
        f"Average Inference Time per Sample: "
        f"{average_inference_time:.6f} seconds\n"
    )

    file.write(
        f"Number of Parameters: "
        f"{total_parameters}\n"
    )

    file.write(
        f"Test Samples: "
        f"{total}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{accuracy_percentage:.2f}%\n"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n==============================================")
print("           DEBERTA TRAINING COMPLETED")
print("==============================================")

print(
    "\nDeBERTa evaluation results saved successfully."
)

print(
    "Saved at:",
    result_file
)

print(
    "\nDeBERTa computational cost results saved successfully."
)

print(
    "Saved at:",
    computational_file
)
