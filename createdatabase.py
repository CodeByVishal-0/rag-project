from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# Loading PDF
data = PyPDFLoader("py.pdf")
docs = data.load()

# Creating Chunks of the data
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Storing Embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma-db"
)
vectorstore.persist()
print("✅ Database created successfully!")
