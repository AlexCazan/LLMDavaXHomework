import os
from openai import OpenAI
from PyPDF2 import PdfReader
import openai


class FileUploader:
    def __init__(self, api_key: str, data_dir: str, chromaDB):
        self.client = OpenAI(api_key=api_key)
        self.data_dir = data_dir
        self.chromaDB = chromaDB

    def structure_file(self):
        file_name = os.listdir(self.data_dir)
        file_path = os.path.join(self.data_dir, file_name[0])
        file = PdfReader(file_path)
        full_text = ""
        for page in file.pages:
            full_text += page.extract_text()

        raw_entries = full_text.split("Title: ")[1:]
        books = []
        for book in raw_entries:
            lines = book.strip().splitlines()
            title = lines[0].strip()
            summary = " ".join(line.strip() for line in lines[1:])
            books.append({"title": title, "summary": summary})
        return books

    @staticmethod
    def embed(text: str) -> list[float]:
        response = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    def upload_files(self):
        collection = self.chromaDB.get_or_create_collection(name="book_summaries")
        books = self.structure_file()
        for book in books:
            text = book["summary"]
            vector = self.embed(text)
            collection.add(
                ids=[book["title"]],
                documents=[book["summary"]],
                metadatas=[{"title": book["title"]}],
                embeddings=[vector],
            )
        return collection
