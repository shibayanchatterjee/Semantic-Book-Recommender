# Semantic-Book-Recommender

A semantic book recommendation system using vector search and language models.

## Features

- Loads and processes book descriptions
- Generates embeddings using OpenAI
- Stores and searches vectors with Chroma
- Returns semantically relevant book recommendations

## Requirements

- Python 3.8+
- pandas
- langchain
- langchain-community
- langchain-openai
- langchain-chroma
- python-dotenv

## Project Structure

```
semantic-book-recommender/
├── README.md
├── requirements.txt
├── .env
├── data/
│   ├── books.csv
│   └── embeddings/
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   ├── update_embeddings.py
│   ├── view_embeddings.py
│   └── delete_embeddings.py
└── main.py
```

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```
3. Create a `.env` file in the root directorywith your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the script:
   ```bash
    python -m src.recommender
    ```
5. Follow the prompts to get book recommendations based on your input.
6. To add new books, modify the `books.csv` file in the `data` directory and rerun the script.
7. To update the embeddings, run the `update_embeddings.py` script:
   ```bash
   python -m src.update_embeddings
   ```
8. To view the current embeddings, run the `view_embeddings.py` script:
   ```bash
    python -m src.view_embeddings
    ```
9. To delete the embeddings, run the `delete_embeddings.py` script:
```bash
    python -m src.delete_embeddings
```
10. To view the current embeddings, run the `view_embeddings.py` script:
```bash
    python -m src.view_embeddings
```
11. To delete the embeddings, run the `delete_embeddings.py` script:
```bash
    python -m src.delete_embeddings
```
12. To view the current embeddings, run the `view_embeddings.py` script:
```bash
    python -m src.view_embeddings
```

## License
MIT License

