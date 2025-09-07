from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
reader = PdfReader("test.pdf")
text = "".join([page.extract_text() for page in reader.pages])

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

# Load embeddings model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert each chunk into vector
vectors = embedder.encode(chunks)

print(f"Generated {len(vectors)} embeddings")
print("Vector shape of first chunk:", vectors[0].shape)
