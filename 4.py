import torch
import pandas as pd
import os

from transformers import AutoTokenizer

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
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

print(
    "Tokenizer loaded successfully."
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
# Initialize model
# --------------------------------------------------

model = SimpleSLM(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=64,
    num_classes=2,
    max_length=64
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
    "Trained ALBERT-inspired prototype "
    "loaded successfully."
)


# --------------------------------------------------
# Function to split paragraph into sentences
# --------------------------------------------------

def split_sentences(text):

    sentences = text.split(".")

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


# --------------------------------------------------
# Function to collect paragraph
# --------------------------------------------------

def collect_text():

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return " ".join(lines)


# --------------------------------------------------
# AI TEXT DETECTOR
# --------------------------------------------------

print("\n==============================================")
print("          AI TEXT DETECTOR")
print("==============================================")


while True:

    print("\nPaste your complete paragraph below.")

    print(
        "You can paste multiple sentences "
        "or multiple lines."
    )

    print(
        "After pasting, type END on a new line "
        "and press Enter."
    )

    print("----------------------------------------------")

    print("Provide text for scan:")

    text = collect_text()


    # --------------------------------------------------
    # Check empty input
    # --------------------------------------------------

    if not text.strip():

        print(
            "\nNo text was provided."
        )

        continue


    # --------------------------------------------------
    # Split paragraph into sentences
    # --------------------------------------------------

    sentences = split_sentences(
        text
    )


    print("\n==============================================")
    print("         SENTENCE ANALYSIS")
    print("==============================================")


    print(
        f"\nTotal sentences detected: "
        f"{len(sentences)}"
    )


    # --------------------------------------------------
    # Store prediction results
    # --------------------------------------------------

    results = []

    total_ai_probability = 0.0
    total_human_probability = 0.0

    ai_sentence_count = 0
    human_sentence_count = 0


    # --------------------------------------------------
    # Analyze every sentence
    # --------------------------------------------------

    for index, sentence in enumerate(
        sentences,
        start=1
    ):

        # --------------------------------------------------
        # Tokenize one sentence
        # --------------------------------------------------

        encoding = tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )


        input_ids = encoding[
            "input_ids"
        ].to(device)

        attention_mask = encoding[
            "attention_mask"
        ].to(device)


        # --------------------------------------------------
        # Model prediction
        # --------------------------------------------------

        with torch.no_grad():

            outputs = model(
                input_ids,
                attention_mask
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


        human_probability = (
            probabilities[0][0].item()
        )

        ai_probability = (
            probabilities[0][1].item()
        )


        # --------------------------------------------------
        # Determine classification
        # --------------------------------------------------

        if ai_probability > human_probability:

            classification = "AI"

            ai_sentence_count += 1

        else:

            classification = "HUMAN"

            human_sentence_count += 1


        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        results.append({
            "Sentence": sentence,
            "Human Probability": human_probability,
            "AI Probability": ai_probability,
            "Classification": classification
        })


        # --------------------------------------------------
        # Add probabilities
        # --------------------------------------------------

        total_human_probability += (
            human_probability
        )

        total_ai_probability += (
            ai_probability
        )


        # --------------------------------------------------
        # Display sentence result
        # --------------------------------------------------

        print(
            f"\nSentence {index}"
        )

        print(
            "----------------------------------------------"
        )

        print(
            sentence
        )

        print(
            f"Human Probability: "
            f"{human_probability * 100:.2f}%"
        )

        print(
            f"AI Probability: "
            f"{ai_probability * 100:.2f}%"
        )

        print(
            f"Classification: "
            f"{classification}"
        )


    # --------------------------------------------------
    # Calculate overall probabilities
    # --------------------------------------------------

    number_of_sentences = len(
        sentences
    )

    overall_human_probability = (
        total_human_probability
        /
        number_of_sentences
    )

    overall_ai_probability = (
        total_ai_probability
        /
        number_of_sentences
    )


    # --------------------------------------------------
    # Determine final paragraph classification
    # --------------------------------------------------

    if overall_ai_probability > overall_human_probability:

        final_classification = (
            "LIKELY AI-WRITTEN"
        )

    else:

        final_classification = (
            "LIKELY HUMAN-WRITTEN"
        )


    # --------------------------------------------------
    # Display final result
    # --------------------------------------------------

    print("\n==============================================")
    print("           OVERALL SCAN RESULT")
    print("==============================================")


    print(
        f"\nTotal Sentences: "
        f"{number_of_sentences}"
    )

    print(
        f"AI-classified sentences: "
        f"{ai_sentence_count}"
    )

    print(
        f"Human-classified sentences: "
        f"{human_sentence_count}"
    )


    print(
        f"\nAverage Human Probability: "
        f"{overall_human_probability * 100:.2f}%"
    )

    print(
        f"Average AI Probability: "
        f"{overall_ai_probability * 100:.2f}%"
    )


    print(
        f"\nFinal Classification: "
        f"{final_classification}"
    )


    print("\n==============================================")


    # --------------------------------------------------
    # Ask for another paragraph
    # --------------------------------------------------

    print(
        "\nYou can enter another paragraph."
    )

    print(
        "Type END after the paragraph again."
    )

    print("----------------------------------------------")
