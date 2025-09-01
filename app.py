import json
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from file_uploader import FileUploader


# Tool function: retrieve full summary by exact title using the books list
def get_summary_by_title(title: str, books: list[dict]) -> str:
    """
    Returns the full summary for an exact book title from the list of books.
    """
    for book in books:
        if book.get("title") == title:
            return book.get("summary", "")
    return f"Summary for '{title}' not found."


GET_SUMMARY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the full summary for a book by its exact title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The exact title of the book.",
                    }
                },
                "required": ["title"],
            },
        },
    }
]

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=API_KEY)


def retrieve_candidates(user_query: str, collection, uploader, k: int = 5) -> list[str]:
    """
    Embed user_query with your uploader and ask Chroma for top-k metadata.
    Returns a list of unique titles (case-insensitive dedupe).
    """
    q_vec = uploader.embed(user_query)
    res = collection.query(query_embeddings=[q_vec], n_results=k)
    titles = []
    seen = set()
    for meta in (res.get("metadatas") or [[]])[0]:
        title = (meta or {}).get("title", "")
        key = title.strip().lower()
        if title and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles


def chat_with_rag_and_tool(
    user_text: str, books: list[dict], collection, uploader
) -> str:
    # 1) Retrieve top-k titles from Chroma
    candidates = retrieve_candidates(user_text, collection, uploader, k=5)

    # 2) Guardrail: if nothing retrieved, short fallback
    if not candidates:
        return "I couldn't find a matching book. Try a different query or a more specific title."

    # 3) Build messages that constrain the model
    system_prompt = (
        "You are a helpful book assistant.\n"
        "- You MUST pick at most ONE title from the provided candidate list.\n"
        "- If the user clearly asked for an exact title in the list, call the tool `get_summary_by_title` with that title.\n"
        "- If the query is thematic/vague, choose the BEST MATCH from candidates and call the tool for that title.\n"
        "- NEVER invent titles; only use those in candidates.\n"
        "- If none of the candidates seem relevant, say so and ask the user for a clearer title."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {
            "role": "system",
            "name": "retrieval_context",
            "content": json.dumps({"candidates": candidates}),
        },
    ]

    # 4) First pass: let the model decide to call the tool
    first = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=GET_SUMMARY_TOOL,
        tool_choice="auto",
    )

    choice = first.choices[0]
    tool_calls = getattr(choice.message, "tool_calls", None)

    if tool_calls:
        # 5) Execute tool(s) locally
        for call in tool_calls:
            if call.function.name == "get_summary_by_title":
                args = json.loads(call.function.arguments or "{}")
                title = args.get("title", "")

                # Defensive: allow only from candidates
                if title not in candidates:
                    # If the model proposed something outside the list, force the top candidate instead
                    title = candidates[0]

                result = get_summary_by_title(title, books)

                messages.append({"role": "assistant", "tool_calls": [call]})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": "get_summary_by_title",
                        "content": result,
                    }
                )

        # 6) Second pass: final answer that uses the tool output
        final = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return final.choices[0].message.content

    # 7) If the model didn’t call the tool, do a pragmatic fallback:
    # call the tool for the top candidate and present it.
    top_title = candidates[0]
    result = get_summary_by_title(top_title, books)
    return f"Best match: **{top_title}**\n\n{result}"


# Streamlit App
def main():
    chroma_client = chromadb.Client(
        Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False,
        )
    )

    # Read and structure PDF file
    uploader = FileUploader(api_key=API_KEY, data_dir="files", chromaDB=chroma_client)
    books = uploader.structure_file()  # list of {'title','summary'}

    # Upload embeddings to ChromaDB
    collection = uploader.upload_files()

    # Streamlit UI configuration
    st.set_page_config(page_title="Book Recommendation Chatbot", layout="centered")
    st.title("🤖 Book Recommendation Chatbot with RAG, GPT")

    user_question = st.text_input(
        "How can I help you with a book recommendation?",
        placeholder="E.g.: I want a book about friendship and magic...",
    )

    if user_question:
        reply = chat_with_rag_and_tool(
            user_text=user_question,
            books=books,
            collection=collection,
            uploader=uploader,
        )
        st.write(reply)

    # Display the total number of stored books
    try:
        total = len(collection.get()["ids"])
        st.caption(f"📚 Total books available: {total}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
