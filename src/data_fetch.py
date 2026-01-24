"""
APERA PoC - arXiv Data Fetcher
Fetches AI ethics papers from arXiv with progress tracking.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from urllib.parse import quote


class ArxivFetcher:
    """Fetch and download arXiv papers for the APERA PoC."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize fetcher with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.csv"
        
    def build_query(
        self, 
        search_term: str = "AI ethics",
        categories: List[str] = None,
        max_results: int = 100
    ) -> str:
        """Build arXiv API query URL."""
        if categories is None:
            categories = ["cs.CL", "cs.CY"]
            
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = f'({cat_query}) AND all:"{search_term}"'
        
        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        query_string = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        return f"{self.BASE_URL}?{query_string}"
    
    def parse_response(self, xml_content: str) -> List[Dict]:
        """Parse arXiv API XML response into paper metadata."""
        root = ET.fromstring(xml_content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                'id': entry.find('atom:id', ns).text.split('/')[-1],
                'title': entry.find('atom:title', ns).text.strip(),
                'summary': entry.find('atom:summary', ns).text.strip(),
                'published': entry.find('atom:published', ns).text,
                'authors': [
                    author.find('atom:name', ns).text 
                    for author in entry.findall('atom:author', ns)
                ],
                'pdf_url': None
            }
            
            # Get PDF link
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
                    break
            
            papers.append(paper)
        
        return papers
    
    def download_pdf(self, url: str, arxiv_id: str) -> bool:
        """Download a single PDF with retry logic."""
        pdf_path = self.data_dir / f"{arxiv_id}.pdf"
        
        # Skip if already downloaded
        if pdf_path.exists():
            return True
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            time.sleep(3)  # arXiv rate limit: 1 request per 3 seconds
            return True
            
        except Exception as e:
            print(f"\n⚠️  Failed to download {arxiv_id}: {e}")
            return False
    
    def fetch_papers(
        self, 
        search_term: str = "AI ethics",
        categories: List[str] = None,
        max_results: int = 100,
        download_pdfs: bool = True
    ) -> List[Dict]:
        """Main fetch pipeline."""
        if categories is None:
            categories = ["cs.CL", "cs.CY"]
            
        print(f"🔍 Searching arXiv for: '{search_term}'")
        print(f"📚 Categories: {', '.join(categories)}")
        print(f"📊 Max results: {max_results}\n")
        
        # Build and execute query
        query_url = self.build_query(search_term, categories, max_results)
        
        try:
            response = requests.get(query_url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return []
        
        # Parse papers
        papers = self.parse_response(response.text)
        print(f"✅ Found {len(papers)} papers\n")
        
        # Download PDFs if requested
        if download_pdfs:
            print("📥 Downloading PDFs...")
            success_count = 0
            
            for paper in tqdm(papers, desc="Progress"):
                if paper['pdf_url']:
                    if self.download_pdf(paper['pdf_url'], paper['id']):
                        success_count += 1
            
            print(f"\n✅ Downloaded {success_count}/{len(papers)} PDFs")
        
        # Save metadata
        self._save_metadata(papers)
        
        return papers
    
    def _save_metadata(self, papers: List[Dict]):
        """Save paper metadata to CSV."""
        import csv
        
        with open(self.metadata_file, 'w', newline='', encoding='utf-8') as f:
            if not papers:
                return
            
            # Convert list fields to strings
            clean_papers = []
            for paper in papers:
                clean = paper.copy()
                clean['authors'] = '; '.join(paper['authors'])
                clean_papers.append(clean)
                
            writer = csv.DictWriter(f, fieldnames=clean_papers[0].keys())
            writer.writeheader()
            writer.writerows(clean_papers)
        
        print(f"💾 Metadata saved to {self.metadata_file}")


# Example usage
if __name__ == "__main__":
    fetcher = ArxivFetcher(data_dir="data/raw")
    
    # Start with 10 papers for testing
    papers = fetcher.fetch_papers(
        search_term="AI ethics",
        categories=["cs.CL", "cs.CY", "cs.AI"],
        max_results=100,  
        download_pdfs=True
    )
    
    print(f"\n🎉 Complete! Check the 'data/raw' folder for PDFs.")
    if papers:
        print(f"📊 Example paper: {papers[0]['title']}")
