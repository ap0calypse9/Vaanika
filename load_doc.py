from pypdf import PdfReader


reader = PdfReader("test.pdf")  
text = "".join([page.extract_text() for page in reader.pages])

print("=== Raw Document Text ===")
print(text[:1000])   # printing first 1000 chars only
