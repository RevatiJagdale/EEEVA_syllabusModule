import pdfplumber
import pandas as pd
import camelot
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io


def pdf_to_markdown(pdf_file, output_md="output.md"):
    """
    Convert a PDF (text/table-rich or scanned) into Markdown.
    ✅ Processes every page
    ✅ Uses pdfplumber first
    ✅ Uses Camelot for tables if needed
    ✅ Falls back to OCR for scanned pages
    """
    content = []

    # --- Process each page with pdfplumber ---
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"[INFO] Processing page {page_num}/{total_pages}")
            page_content = []

            # --- Text extraction ---
            text = page.extract_text()
            if text:
                page_content.append(f"## Page {page_num}\n\n{text}\n")

            # --- Table extraction with pdfplumber ---
            tables = page.extract_tables()
            for t in tables:
                try:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    page_content.append(df.to_markdown(index=False))
                except Exception:
                    pass

            # --- If no tables found, try Camelot ---
            if not tables:
                try:
                    camelot_tables = camelot.read_pdf(pdf_file, pages=str(page_num))
                    for i, table in enumerate(camelot_tables):
                        page_content.append(f"### Camelot Table {i+1} (Page {page_num})\n\n")
                        page_content.append(table.df.to_markdown(index=False))
                except Exception as e:
                    print(f"[WARN] Camelot failed on page {page_num}: {e}")

            # --- If nothing extracted at all, fallback to OCR ---
            if not page_content:
                print(f"[WARN] Page {page_num} empty → using OCR")
                fitz_page = fitz.open(pdf_file)[page_num - 1]
                pix = fitz_page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    page_content.append(f"## Page {page_num} (OCR)\n\n{text}\n")

            if page_content:
                content.append("\n\n".join(page_content))

    # --- Save everything to Markdown ---
    if content:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(content))
        print(f"[SUCCESS] Full PDF extracted → {output_md}")
    else:
        print("[FAIL] No content extracted at all.")


if __name__ == "__main__":
    PDF_FILE = "data/22-26 ECE AIML Structure and Syllabus.pdf"
    pdf_to_markdown(PDF_FILE, "syllabus.md")
