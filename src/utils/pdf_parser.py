"""
APERA PoC - PDF Text Extraction
Extracts text from PDF files with error handling and progress tracking.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDF files robustly."""
    
    def __init__(self, min_text_length: int = 100):
        """
        Initialize PDF parser.
        
        Args:
            min_text_length: Minimum text length to consider a page valid
        """
        self.min_text_length = min_text_length
        
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with 'text', 'num_pages', 'metadata', or None if failed
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'pages': len(doc)
            }
            
            # Extract text from all pages
            full_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean up text
                text = self._clean_text(text)
                
                if len(text) >= self.min_text_length:
                    full_text.append(text)
            
            doc.close()
            
            if not full_text:
                logger.warning(f"No valid text extracted from {pdf_path.name}")
                return None
            
            return {
                'text': '\n\n'.join(full_text),
                'num_pages': len(doc),
                'metadata': metadata,
                'file_name': pdf_path.name
            }
            
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path.name}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Null bytes
        text = text.replace('\ufffd', '')  # Replacement character
        
        return text.strip()
    
    def extract_from_directory(
        self, 
        pdf_dir: Path,
        pattern: str = "*.pdf"
    ) -> List[Dict]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDFs
            pattern: Glob pattern for PDF files
            
        Returns:
            List of extracted document dicts
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        failed = 0
        
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            result = self.extract_text_from_pdf(pdf_path)
            if result:
                documents.append(result)
            else:
                failed += 1
        
        logger.info(f"Successfully parsed {len(documents)}/{len(pdf_files)} PDFs")
        if failed > 0:
            logger.warning(f"Failed to parse {failed} PDFs")
        
        return documents
    
    def extract_with_arxiv_id(self, pdf_path: Path) -> Optional[Dict]:
        """
        Extract text and preserve arXiv ID from filename.
        
        Args:
            pdf_path: Path to PDF (e.g., '2401.12345.pdf')
            
        Returns:
            Dict with text and arxiv_id
        """
        result = self.extract_text_from_pdf(pdf_path)
        
        if result:
            # Extract arXiv ID from filename (e.g., '2401.12345.pdf' -> '2401.12345')
            arxiv_id = pdf_path.stem
            result['arxiv_id'] = arxiv_id
        
        return result


# Example usage and testing
if __name__ == "__main__":
    parser = PDFParser()
    
    # Test on data/raw directory
    documents = parser.extract_from_directory(Path("data/raw"))
    
    if documents:
        print(f"\n✅ Parsed {len(documents)} papers")
        print(f"\n📄 First document preview:")
        print(f"File: {documents[0]['file_name']}")
        print(f"Pages: {documents[0]['num_pages']}")
        print(f"Text length: {len(documents[0]['text'])} characters")
        print(f"Preview: {documents[0]['text'][:200]}...")
        
        # Save extracted text for inspection
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        for doc in documents:
            output_file = output_dir / f"{doc['file_name'].replace('.pdf', '.txt')}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(doc['text'])
        
        print(f"\n💾 Saved extracted text to data/processed/")
    else:
        print("❌ No documents parsed. Check data/raw/ directory.")