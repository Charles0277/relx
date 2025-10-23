# Python Text Summarizer with Named Entity Recognition

This is a command-line tool that generates a concise summary of one or more text files and extracts named entities like people, organizations, and dates.

## Features

- **Text Summarization**: Creates a short, readable summary of long texts.
- **Named Entity Recognition (NER)**: Identifies and categorizes entities such as:
  - People (PERSON)
  - Organizations (ORG)
  - Locations (GPE)
  - Dates (DATE)
- **Multi-File Support**: Can process multiple text files at once.

## Technical Choices

- **Summarization**: I used the `transformers` library from Hugging Face with the `facebook/bart-large-cnn` model. This pre-trained model is excellent for generating high-quality, abstractive summaries that are fluent and human-like.
- **Named Entity Recognition**: I chose `spacy` with its `en_core_web_sm` model for NER. It is a powerful and efficient library for various NLP tasks, and it provides a straightforward way to extract entities from text.
- **Command-Line Interface**: The script uses Python's built-in `argparse` module to provide a simple and effective command-line interface.

## Setup and Usage

### 1. Prerequisites

- Python 3.7+

### 2. Installation

1.  **Clone the repository (or download the files):**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the spacy model:**
    This model is needed for named entity recognition.
    ```bash
    python -m spacy download en_core_web_sm
    ```

### 3. Running the Script

To summarize one or more text files, run the script from your terminal:

```bash
python summarizer.py path/to/file1.txt path/to/file2.txt
```

#### Example

Using the sample article provided:

```bash
python summarizer.py sample_texts/article1.txt
```

## Limitations

- **Model Size**: The `facebook/bart-large-cnn` model is large, and the first time you run the script, it will download several hundred megabytes of data.
- **Input Length**: The summarization model has a maximum input length of 1024 tokens. For texts longer than this, the script will truncate the input, which may result in an incomplete summary.
- **File Types**: The script currently only supports plain text (.txt) files.

## Ideas for Improvement

- **Support for More File Types**: Add support for other file formats like PDF, DOCX, and web pages.
- **Chunking for Long Texts**: For documents that exceed the model's token limit, implement a chunking strategy to summarize parts of the text and then combine the summaries.
- **GUI**: Develop a simple graphical user interface (GUI) to make the tool more user-friendly.
- **API Integration**: Wrap the functionality in a REST API to allow other applications to use the summarization and NER capabilities.
