from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load PDF
reader = PdfReader("test.pdf")
text = "".join([page.extract_text() for page in reader.pages])

# Split the content into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Wrap chunks in Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Embeddings model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma
vectordb = Chroma.from_documents(docs, embedding=embedder)


# Test similarity search
query = "What is this document about?"
results = vectordb.similarity_search(query, k=2)

print("=== Top results ===")
for r in results:
    print(r.page_content[:200], "\n---")
