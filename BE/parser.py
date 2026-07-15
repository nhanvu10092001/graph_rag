"""Document text extraction utility supporting .txt, .md, and .pdf files."""

import logging
from typing import Optional

logger = logging.getLogger("BE.parser")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text content from PDF file bytes using pypdfium2."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_bytes)
        text = ""
        for page in pdf:
            textpage = page.get_textpage()
            text += textpage.get_text_bounded() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extracts plain text content from various file formats based on extension."""
    ext = filename.lower().split(".")[-1]
    
    if ext in ["txt", "md"]:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return file_bytes.decode("latin-1")
            except Exception as e:
                raise ValueError(f"Failed to decode text file: {str(e)}")
                
    elif ext == "pdf":
        return extract_text_from_pdf(file_bytes)
        
    else:
        raise ValueError(f"Unsupported file format: '.{ext}'. Supported formats: .txt, .md, .pdf")
