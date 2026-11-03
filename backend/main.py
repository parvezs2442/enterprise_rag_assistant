
import os
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Google API key not found in .env")


#document loader

pdf_path = "data/notes.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"\n Pages Loaded: {len(documents)}")