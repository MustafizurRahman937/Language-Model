import csv
import os


# --------------------------------------------------
# Locate the current directory
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


# --------------------------------------------------
# Function to separate text into sentences
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
# Function to collect multiple lines of text
# --------------------------------------------------

def collect_text():

    lines = []

    while True:

        line = input()

        # END means the user has finished pasting
        if line.strip().upper() == "END":
            break

        lines.append(line)

    return " ".join(lines)


# --------------------------------------------------
# Function to add data to CSV
# --------------------------------------------------

def add_to_csv(file_path, sentences, label):

    file_exists = os.path.exists(file_path)

    file_is_empty = (
        not file_exists
        or os.path.getsize(file_path) == 0
    )

    # Append mode keeps existing data
    with open(
        file_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Add header only for a new/empty file
        if file_is_empty:

            writer.writerow([
                "text",
                "label"
            ])

        for sentence in sentences:

            writer.writerow([
                sentence,
                label
            ])


# --------------------------------------------------
# HUMAN TEXT DATASET
# --------------------------------------------------

print("==============================================")
print("       HUMAN TEXT DATASET GENERATION")
print("==============================================")

print("\nPaste Human-written text below.")
print("You can paste multiple paragraphs or lines.")
print("When finished, type END on a new line.")
print("----------------------------------------------")

human_text = collect_text()

human_sentences = split_sentences(
    human_text
)

add_to_csv(
    human_file,
    human_sentences,
    0
)

print("\n----------------------------------------------")

print(
    f"{len(human_sentences)} new human-written "
    "sentences detected."
)

print(
    "Human dataset updated successfully."
)

print(
    "Saved at:",
    human_file
)


# --------------------------------------------------
# AI TEXT DATASET
# --------------------------------------------------

print("\n==============================================")
print("         AI TEXT DATASET GENERATION")
print("==============================================")

print("\nNow paste AI-written text below.")
print("You can paste multiple paragraphs or lines.")
print("When finished, type END on a new line.")
print("----------------------------------------------")

ai_text = collect_text()

ai_sentences = split_sentences(
    ai_text
)

add_to_csv(
    ai_file,
    ai_sentences,
    1
)

print("\n----------------------------------------------")

print(
    f"{len(ai_sentences)} new AI-written "
    "sentences detected."
)

print(
    "AI dataset updated successfully."
)

print(
    "Saved at:",
    ai_file
)


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\n==============================================")
print("          DATASET GENERATION COMPLETED")
print("==============================================")

print(
    f"New Human sentences added: "
    f"{len(human_sentences)}"
)

print(
    f"New AI sentences added: "
    f"{len(ai_sentences)}"
)

print("\nHuman.csv and AI.csv have been updated.")