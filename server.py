from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, shutil, tempfile, requests
from bs4 import BeautifulSoup

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

embeddings  = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
splitter    = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
model       = ChatGroq(model="llama-3.3-70b-versatile")
vectorstore = None
retriever   = None

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.
If the answer is not present in the context, say: "I could not find the answer in the document." """),
    ("human", "context:{context}\n\nQuestion:{question}")
])

def build_retriever(docs):
    global vectorstore, retriever
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma-db"
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

# Load existing DB on startup
if os.path.exists("chroma-db"):
    try:
        vectorstore = Chroma(persist_directory="chroma-db", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
        print("✅ Loaded existing chroma-db")
    except Exception as e:
        print(f"⚠️ Could not load chroma-db: {e}")

# ── Status ──
@app.get("/status")
async def status():
    return {"ready": retriever is not None}

# ── Upload PDF ──
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        build_retriever(docs)
        os.unlink(tmp_path)
        return {"status": "ok", "message": f"✅ '{file.filename}' ingested — {len(docs)} page(s) ready."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ── Ingest URL — robust scraper ──
class URLRequest(BaseModel):
    url: str

def scrape_url(url: str) -> list[Document]:
    """
    Robust URL scraper using requests + BeautifulSoup.
    Handles dynamic sites better than WebBaseLoader.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise: scripts, styles, nav, footer, ads
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Try to get main content area first
    main = (
        soup.find("article") or
        soup.find("main") or
        soup.find(id="content") or
        soup.find(class_="content") or
        soup.find(class_="post-content") or
        soup.find(class_="entry-content") or
        soup.body
    )

    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    # Clean up excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    if not clean_text:
        raise ValueError("No readable text found on this page.")

    return [Document(page_content=clean_text, metadata={"source": url})]


@app.post("/ingest-url")
async def ingest_url(req: URLRequest):
    try:
        docs = scrape_url(req.url)
        total_chars = sum(len(d.page_content) for d in docs)
        build_retriever(docs)
        return {
            "status": "ok",
            "message": f"✅ URL ingested — {total_chars:,} characters extracted and indexed."
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "❌ Could not connect to the URL. Check if it's accessible."}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "❌ Request timed out. The site took too long to respond."}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "message": f"❌ HTTP error: {e}"}
    except ValueError as e:
        return {"status": "error", "message": f"❌ {e}"}
    except Exception as e:
        return {"status": "error", "message": f"❌ Unexpected error: {str(e)}"}

# ── Chat ──
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    if retriever is None:
        return {"answer": "⚠️ No document loaded. Upload a PDF or paste a URL first."}
    try:
        docs = retriever.invoke(req.question)
        context = "\n\n".join([d.page_content for d in docs])
        response = model.invoke(prompt.invoke({"context": context, "question": req.question}))
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}