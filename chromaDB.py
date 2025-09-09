from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM   

# Load PDF
reader = PdfReader("test.pdf")
text = "".join([page.extract_text() for page in reader.pages])

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Wrap chunks in Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Embedding model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma
vectordb = Chroma.from_documents(docs, embedding=embedder, persist_directory="./chroma_store")

# Local LLM (phi via Ollama)
llm = OllamaLLM(model="phi")

print(" Document loaded. You can now ask questions (type 'exit' to quit).")

while True:
    query = input("\n Your question: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    # Retrieve context
    results = vectordb.similarity_search(query, k=3)
    context = "\n\n".join([r.page_content for r in results])

    # Build prompt
    prompt = f"""Use the following document context to answer the question:

Context:
{context}

Question: {query}

Answer clearly and concisely:
"""

    # Run through Ollama
    response = llm.invoke(prompt)

    print("\n Answer:")
    print(response)
