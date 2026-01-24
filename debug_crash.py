import os
import fitz # PyMuPDF
from src.utils.chunking import PDFProcessor

print("1. Testing Imports...")
processor = PDFProcessor()
print("✅ Imports OK")

data_dir = "data/raw"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pdf")])
print(f"2. Found {len(files)} PDFs. Testing the first one: {files[0]}")

path = os.path.join(data_dir, files[0])
try:
    print(f"   Opening {path}...")
    doc = fitz.open(path)
    print(f"   Page count: {len(doc)}")
    text = doc[0].get_text()
    print(f"   Extracted {len(text)} chars from page 1.")
    print("✅ PyMuPDF is working.")
except Exception as e:
    print(f"❌ CRASHED on PDF loading: {e}")

print("3. Testing Chunking...")
try:
    chunks = processor.load_and_chunk(path)
    print(f"✅ Chunking OK. Generated {len(chunks)} chunks.")
except Exception as e:
    print(f"❌ CRASHED on Chunking logic: {e}")
