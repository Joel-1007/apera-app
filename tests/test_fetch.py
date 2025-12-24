"""Tests for arXiv fetcher."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetch import ArxivFetcher


def test_fetcher_initialization():
    """Test fetcher creates data directory."""
    fetcher = ArxivFetcher(data_dir="data/test")
    assert fetcher.data_dir.exists()


def test_build_query():
    """Test query URL construction."""
    fetcher = ArxivFetcher()
    url = fetcher.build_query(
        search_term="AI ethics",
        categories=["cs.CL"],
        max_results=10
    )
    assert "export.arxiv.org" in url
    assert "AI%20ethics" in url
    assert "max_results=10" in url


def test_parse_response():
    """Test XML parsing with sample data."""
    fetcher = ArxivFetcher()
    
    sample_xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>Test Paper</title>
            <summary>Test summary</summary>
            <published>2024-01-01T00:00:00Z</published>
            <author><name>Test Author</name></author>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
        </entry>
    </feed>'''
    
    papers = fetcher.parse_response(sample_xml)
    assert len(papers) == 1
    assert papers[0]['title'] == "Test Paper"
    assert papers[0]['id'] == "1234.5678v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
