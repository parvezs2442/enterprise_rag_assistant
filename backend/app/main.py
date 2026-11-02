from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from app.loaders.loader import load_pdf

# 1. .env file se GOOGLE_API_KEY load karna
load_dotenv()


#Loader 

def main():

    documents = load_pdf()

    print("\nFirst Page Content:\n")

    print(documents[0].page_content)

    print("\nMetadata:\n")

    print(documents[0].metadata)


if __name__ == "__main__":
    main()