from pypdf import PdfReader


reader = PdfReader("test.pdf")  
text = ""

for page in reader.pages:
    text += page.extract_text()

print("=== Raw Document Text ===")
print(text[:1000])   # printing  first 1000 chars only
