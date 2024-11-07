# Text Analysis for Term Frequency (TF) and Document Frequency (DF)

This project provides a Python-based solution for calculating Term Frequency (TF) and Document Frequency (DF) from a set of text documents. This approach allows efficient frequency analysis, which is useful for tasks like keyword extraction, similarity search, and other text-based analyses.

## Project Structure

- **Data Ingestion**: Load and preprocess a collection of documents, where each document contains fields like `name` and `description`.
- **TF Calculation**: Calculates Term Frequency (TF) for each word in each document.
- **DF Calculation**: Computes Document Frequency (DF) for each word across the entire document set.
- **TF-IDF Calculation**: Combines TF and DF values to calculate TF-IDF scores for more insightful frequency analysis.

## Requirements

- Python 3.x
- `pandas` for data handling (install using `pip install pandas`)

## Setup and Usage

1. **Install Dependencies**: Ensure `pandas` is installed:
   ```bash
   pip install pandas
   ```
2. **Run the Script**:
   - Load documents from a CSV file or list.
   - Preprocess the text, calculate TF, DF, and TF-IDF scores.

## Code Overview

### Preprocess Data
```python
def preprocess_text(text):
    # Tokenizes and lowercases text
    ...
```

### Calculate Term Frequency (TF)
```python
# Define a function to calculate TF
def calculate_tf(documents):
    ...
```

### Calculate Document Frequency (DF)
```python
# Define a function to calculate DF
def calculate_df(documents):
    ...
```

### Calculate TF-IDF
```python
# Use TF and DF to calculate TF-IDF
import math
tf_idf_results = [...]
```

## Example Output

For each document:
```
Document ID: <id>
  Word: <word>, TF-IDF: <score>
```

## Future Improvements

- **MongoDB Integration**: Implement MongoDB for efficient storage, retrieval, and processing of text data, allowing aggregation-based TF and DF calculations.
- **Advanced Text Preprocessing**: Add options for stop-word removal, stemming, and lemmatization.

## License

This project is open-source and available under the [MIT License](LICENSE).
