import torch
import torch.nn as nn
import torch.optim as optim
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
# Load Human and AI datasets
# --------------------------------------------------

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

print("\nCombined dataset generated.")
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
# Prepare text and labels
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
# Display class distribution
# --------------------------------------------------

human_count = labels.count(0)
ai_count = labels.count(1)

print("\nClass Distribution")
print("-------------------------")

print(
    "Human:",
    human_count
)

print(
    "AI:",
    ai_count
)


# --------------------------------------------------
# 80/20 TRAIN-TEST SPLIT
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


print("\nDataset split completed.")

print(
    "Training samples:",
    len(train_texts)
)

print(
    "Testing samples:",
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
# Tokenization
# --------------------------------------------------

print("\nTokenizing text...")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=64
)

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
# Display tokenization example
# --------------------------------------------------

print("\nTokenization Example")
print("-------------------------")

print("Original text:")

print(
    train_texts[0]
)

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


# --------------------------------------------------
# Make PyTorch Dataset
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
# Make training and testing datasets
# --------------------------------------------------

train_dataset = TextDataset(
    train_encodings,
    train_labels
)

test_dataset = TextDataset(
    test_encodings,
    test_labels
)


# --------------------------------------------------
# Make DataLoaders
# --------------------------------------------------

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


# --------------------------------------------------
# Initialize SimpleSLM
# --------------------------------------------------

vocab_size = tokenizer.vocab_size

model = SimpleSLM(
    vocab_size=vocab_size,
    embedding_dim=64,
    num_classes=2,
    max_length=64
)

print(
    "\nALBERT-inspired prototype initialized."
)

print(
    "Vocabulary size:",
    vocab_size
)


# --------------------------------------------------
# Count parameters
# --------------------------------------------------

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Number of parameters:",
    total_parameters
)


# --------------------------------------------------
# Select device
# --------------------------------------------------

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


# --------------------------------------------------
# Class-weighted loss
# --------------------------------------------------

train_human = train_labels.count(0)
train_ai = train_labels.count(1)

total_train = (
    train_human
    +
    train_ai
)

human_weight = (
    total_train
    /
    (2 * train_human)
)

ai_weight = (
    total_train
    /
    (2 * train_ai)
)

class_weights = torch.tensor(
    [
        human_weight,
        ai_weight
    ],
    dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.01
)


# --------------------------------------------------
# Learning-rate scheduler
# --------------------------------------------------

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

epochs = 20

best_training_loss = float("inf")

patience = 5

epochs_without_improvement = 0


print("\nTraining started...")

print(
    "Maximum epochs:",
    epochs
)


# --------------------------------------------------
# Train the model
# --------------------------------------------------

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0


    # --------------------------------------------------
    # Training batches
    # --------------------------------------------------

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


        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs = model(
            input_ids,
            attention_mask
        )


        # --------------------------------------------------
        # Calculate loss
        # --------------------------------------------------

        loss = criterion(
            outputs,
            labels_batch
        )


        # --------------------------------------------------
        # Clear previous gradients
        # --------------------------------------------------

        optimizer.zero_grad()


        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        loss.backward()


        # --------------------------------------------------
        # Gradient clipping
        # --------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # --------------------------------------------------
        # Update model parameters
        # --------------------------------------------------

        optimizer.step()


        # --------------------------------------------------
        # Calculate statistics
        # --------------------------------------------------

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions
            ==
            labels_batch
        ).sum().item()

        total += (
            labels_batch.size(0)
        )


    # --------------------------------------------------
    # Epoch statistics
    # --------------------------------------------------

    average_loss = (
        total_loss
        /
        len(train_loader)
    )

    training_accuracy = (
        correct
        /
        total
    )


    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler.step(
        average_loss
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------
    # Display training results
    # --------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{epochs}]"
    )

    print(
        f"Training Loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{training_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )


    # --------------------------------------------------
    # Save best model
    #
    # We use training loss because there is
    # no validation set in the 80/20 design.
    # --------------------------------------------------

    if (
        average_loss
        <
        best_training_loss
    ):

        best_training_loss = (
            average_loss
        )

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            trained_model_file
        )

        print(
            "Best model saved."
        )

    else:

        epochs_without_improvement += 1

        print(
            "Training loss did not improve."
        )


    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

    if (
        epochs_without_improvement
        >=
        patience
    ):

        print(
            "\nEarly stopping activated."
        )

        break


# --------------------------------------------------
# Load best model
# --------------------------------------------------

print(
    "\nLoading best trained model..."
)

model.load_state_dict(
    torch.load(
        trained_model_file,
        map_location=device
    )
)

model.to(device)


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

model.eval()

correct = 0

total = 0

print(
    "\nPrototype Evaluation"
)

print(
    "-------------------------"
)


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


        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs = model(
            input_ids,
            attention_mask
        )


        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        # --------------------------------------------------
        # Accuracy
        # --------------------------------------------------

        correct += (
            predictions
            ==
            labels_batch
        ).sum().item()

        total += (
            labels_batch.size(0)
        )


# --------------------------------------------------
# Calculate final accuracy
# --------------------------------------------------

accuracy = (
    correct
    /
    total
)


# --------------------------------------------------
# Display final result
# --------------------------------------------------

print(
    "Correct Predictions:",
    correct
)

print(
    "Total Test Samples:",
    total
)

print(
    "Prototype Accuracy:",
    accuracy
)

print(
    "Prototype Accuracy Percentage:",
    f"{accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print(
    "\nTraining completed."
)

print(
    "Best trained model saved successfully."
)

print(
    "Saved model:",
    trained_model_file
)