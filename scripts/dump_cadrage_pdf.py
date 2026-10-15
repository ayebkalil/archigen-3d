import pypdf

reader = pypdf.PdfReader(r"C:\Users\ayebk\Downloads\ArchiGen3D_Cadrage_DeepLearning_Medamine_Benfadhel.pdf")
print(f"Total pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    print(f"\n==================== PAGE {i+1} ====================")
    print(page.extract_text())
