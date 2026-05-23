from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """),
    ("human", "context:{context} Question:{question}")
])

print("=== RAG Terminal Chat ===")
print("1. Load PDF")
print("2. Load from URL")
choice = input("Choose source (1/2): ").strip()

if choice == "1":
    path = input("PDF path: ").strip()
    docs = PyPDFLoader(path).load()
elif choice == "2":
    url = input("Enter URL: ").strip()
    docs = WebBaseLoader(url).load()
else:
    print("Invalid choice"); exit()

chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma-db")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":4,"fetch_k":10,"lambda_mult":0.5})
print("✅ Document loaded! Type your questions (0 to exit)\n")

while True:
    query = input("You : ")
    if query == "0": break
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    response = model.invoke(prompt.invoke({"context": context, "question": query}))
    print("AI :", response.content, "\n")
