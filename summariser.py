import argparse
from transformers import pipeline, AutoTokenizer
import spacy
from collections import defaultdict
import textwrap
import pathlib

# Initialize the summarisation pipeline once
try:
    summariser = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Error initialising summarisation pipeline: {e}")
    summariser = None

def read_files(file_paths):
    """Reads content from a list of text files."""
    content = ""
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content += f.read() + "\n"
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")
    return content.strip()

def summarise_text(text):
    """
    Generates a summary for the given text, handling long inputs by chunking.
    """
    if summariser is None:
        return "Summarisation model not available."

    # The model has a maximum token limit (1024 for BART). We chunk the text to avoid errors.
    max_chunk_length = 1024
    tokenizer = summariser.tokenizer

    # Tokenize the full text
    tokens = tokenizer.encode(text)
    
    summaries = []
    # Process the text in chunks
    for i in range(0, len(tokens), max_chunk_length):
        chunk_tokens = tokens[i:i + max_chunk_length]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Summarise the chunk
        try:
            summary_chunk = summariser(chunk_text, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary_chunk[0]['summary_text'])
        except Exception as e:
            print(f"An error occurred during summarisation of a chunk: {e}")
            
    return " ".join(summaries)

def extract_entities(text):
    """Extracts named entities from the text."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Spacy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        return None
        
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        if ent.text.strip():
            entities[ent.label_].append(ent.text.strip())
    
    # Remove duplicates
    for label in entities:
        entities[label] = sorted(list(set(entities[label])))
        
    return entities

def main():
    parser = argparse.ArgumentParser(description="Summarise text from one or more files and extract named entities.")
    parser.add_argument("files", nargs='+', help="Paths to the text files to be summarised.")
    args = parser.parse_args()

    # 1. Read and combine text from all files
    full_text = read_files(args.files)

    if not full_text:
        print("No content to summarise.")
        return

    print("---" * 10)
    print("Original Text:")
    print("---" * 10)
    print(textwrap.fill(full_text, width=80))
    print("\n")


    # 2. Generate summary
    print("---" * 10)
    print("Summary:")
    print("---" * 10)
    summary = summarise_text(full_text)
    print(textwrap.fill(summary, width=80))
    print("\n")

    # 3. Extract named entities
    print("---" * 10)
    print("Named Entities:")
    print("---" * 10)
    entities = extract_entities(full_text)
    if entities:
        for label, items in entities.items():
            print(f"{label}:")
            for item in items:
                print(f"  - {item}")
    else:
        print("No entities found or Spacy model not loaded.")
    print("\n")

    # 4. Save the summary and entities to a file
    if args.files:
        first_file_path = pathlib.Path(args.files[0])
        output_filename = f"{first_file_path.stem}_summarised.txt"
        output_filepath = first_file_path.parent / output_filename
        
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write("--- Summary ---\n")
                f.write(summary)
                f.write("\n\n--- Named Entities ---\n")
                if entities:
                    for label, items in entities.items():
                        f.write(f"{label}:\n")
                        for item in items:
                            f.write(f"  - {item}\n")
                else:
                    f.write("No entities found or Spacy model not loaded.\n")
            
            print(f"Summary and entities saved to {output_filepath}")
        except Exception as e:
            print(f"Could not save results to file: {e}")
        print("\n")


if __name__ == "__main__":
    main()
