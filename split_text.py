from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
reader = PdfReader("test.pdf")
text = "".join([page.extract_text() for page in reader.pages])

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")
print("=== First chunk ===")
print(chunks[0])
