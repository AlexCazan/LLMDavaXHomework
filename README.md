# Smart Librarian

A simple AI-powered chatbot that recommends books and retrieves summaries based on user queries. It uses GPT-4o and OpenAI embeddings for natural language understanding, combined with ChromaDB for vector search. Users can input thematic or title-based prompts and receive relevant book summaries extracted from a structured PDF file.

## Project Structure

```
.
├── app.py                  # Main Streamlit app
├── file_uploader.py        # File parser + embedding logic
├── files/
│   └── book_summaries.pdf  # Input summaries file (required)
├── README.md
├── LICENSE
├── pytest.ini
```

---

## How to Run

1. **Install dependencies**

```bash
pip install streamlit openai chromadb PyPDF2
```

2. **Set your OpenAI API key**

In `app.py`, update:

```python
API_KEY = "your_openai_api_key_here"
```

3. **Run the app**

```bash
streamlit run app.py
```

> Make sure `book_summaries.pdf` is inside the `files/` folder.
