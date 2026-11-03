
import os
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
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


#text splitting

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks : {len(chunks)}")

#Embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

print("\nEMbedding Model Initialized")


#Qdrant Vector Store

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "rag_documents"

# Check whether collection already exists
client = QdrantClient(path=QDRANT_PATH)

collection_exists = client.collection_exists(
    collection_name=COLLECTION_NAME
)

# Close client before opening another local client
client.close()

if collection_exists:

    print("Existing Qdrant collection found!")

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )

else:

    print("Creating new Qdrant collection...")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Documents stored in Qdrant!")


# 6. RETRIEVER
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
print("Retriever Initialized!")

# 7. LLM INITIALIZATION
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)
print("Gemini LLM Initialized!")


# 8. PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided context.

Do not make up information.

If the answer is not available in the
context, say:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")



#RAG Chain
rag_chain = prompt | llm | StrOutputParser()



print("\n================================")
print("       BASIC RAG ASSISTANT")
print("================================")

while True:

    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("\nGoodbye!")
        break

    if not question:
        continue

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("\nAI: No relevant documents found.")
        continue

    # Build context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Generate answer
    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI:", response)

    # Display source pages
    print("\nSources:")

    for doc in retrieved_docs:

        page_number = doc.metadata.get("page", 0) + 1

        print(f"- Page {page_number}")
