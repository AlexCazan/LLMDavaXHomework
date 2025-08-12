import os
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from file_uploader import FileUploader

FILES_DIR = "files"


def get_summary_by_title(title: str, books) -> str:
    """
    Returnează rezumatul complet pentru un titlu exact din dicționarul local.
    """
    return books.get(title, f"Rezumat pentru '{title}' nu a fost găsit.")


functions = [
    {
        "name": "get_summary_by_title",
        "description": "Returnează rezumatul complet pentru un titlu exact de carte.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Titlul exact al cărții"}
            },
            "required": ["title"],
        },
    }
]


def main():
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    openAI_client = OpenAI(API_KEY)
    chromaDB = chromadb.Client(
        Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False,
        )
    )
    uploader = FileUploader(api_key=API_KEY, data_dir=FILES_DIR, chromaDB=chromaDB)
    books = uploader.structure_file()
    collection = uploader.upload_files()
    st.set_page_config(page_title="Chatbot Recomandare Cărți", layout="centered")
    st.title("🤖 Chatbot Recomandare Cărți cu RAG și GPT")

    user_question = st.text_input(
        "Cum pot ajuta cu o recomandare de carte?",
        placeholder="Ex: Vreau o carte despre prietenie și magie...",
    )
    if user_question:
        # 1) Retrieval: embed query, find top match
        q_vec = uploader.embed(user_question)
        result = collection.query(query_embeddings=[q_vec], n_results=1)
        # Extract recommended title
        try:
            rec_meta = result["metadatas"][0][0]
            rec_title = rec_meta.get("title", "")
        except Exception:
            rec_title = None

        if not rec_title:
            st.warning("Nu am găsit nicio recomandare. Încearcă o altă întrebare.")
        else:
            # 2) Use ChatCompletion with function call to fetch full summary
            messages = [
                {
                    "role": "system",
                    "content": "Ești un asistent prietenos care recomandă cărți.",
                },
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": f"Îți recomand '{rec_title}'."},
            ]
            response = openAI_client.chat.completions.create(
                model="o4-mini",
                messages=messages,
                functions=functions,
                function_call={
                    "name": "get_summary_by_title",
                    "arguments": json.dumps({"title": rec_title}),
                },
            )
            message = response.choices[0].message
            if message.get("function_call"):
                summary = get_summary_by_title(rec_title, books)
                st.subheader(f"Recomandare: {rec_title}")
                st.write(summary)
            else:
                st.error("A apărut o eroare la preluarea rezumatului.")


if __name__ == "__main__":
    main()
