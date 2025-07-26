#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence for Adobe Hackathon Round 1B
Extracts and prioritizes relevant sections based on persona and job-to-be-done
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaDocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer with embedding model"""
        # Use a lightweight sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Keywords for different academic/professional domains
        self.domain_keywords = {
            'research': ['methodology', 'results', 'conclusion', 'abstract', 'literature', 'hypothesis', 'experiment'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'analysis', 'growth', 'investment', 'performance'],
            'technical': ['algorithm', 'implementation', 'system', 'architecture', 'design', 'specification'],
            'educational': ['concept', 'theory', 'principle', 'example', 'practice', 'exercise', 'summary'],
            'financial': ['financial', 'budget', 'cost', 'revenue', 'profit', 'loss', 'investment', 'return']
        }
    
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract sections and subsections from PDF with content"""
        sections = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                current_section = None
                content_buffer = []
                
                for block in blocks:
                    if "lines" in block:
                        block_text = ""
                        max_font_size = 0
                        is_bold = False
                        
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                                max_font_size = max(max_font_size, span["size"])
                                if span["flags"] & 2**4:  # Bold flag
                                    is_bold = True
                            block_text += line_text + " "
                        
                        block_text = block_text.strip()
                        
                        # Check if this looks like a section header
                        if self._is_section_header(block_text, max_font_size, is_bold):
                            # Save previous section if exists
                            if current_section and content_buffer:
                                current_section["content"] = " ".join(content_buffer).strip()
                                sections.append(current_section)
                            
                            # Start new section
                            current_section = {
                                "title": block_text,
                                "page": page_num + 1,
                                "document": os.path.basename(pdf_path),
                                "content": ""
                            }
                            content_buffer = []
                        
                        elif current_section and len(block_text) > 20:
                            # Add to content buffer
                            content_buffer.append(block_text)
                
                # Don't forget the last section
                if current_section and content_buffer:
                    current_section["content"] = " ".join(content_buffer).strip()
                    sections.append(current_section)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {str(e)}")
        
        return sections
    
    def _is_section_header(self, text: str, font_size: float, is_bold: bool) -> bool:
        """Determine if text is likely a section header"""
        # Basic heuristics for section headers
        if len(text) < 5 or len(text) > 100:
            return False
        
        # Common section header patterns
        header_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS
            r'^(Introduction|Abstract|Conclusion|Results|Discussion|Methodology|Background)',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+'
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Font-based detection
        if is_bold and font_size > 12:
            return True
        
        return False
    
    def calculate_relevance_scores(self, sections: List[Dict], persona: str, job: str) -> List[Dict]:
        """Calculate relevance scores for sections based on persona and job"""
        
        # Create query embedding from persona and job
        query_text = f"{persona} {job}"
        query_embedding = self.model.encode([query_text])
        
        # Get embeddings for all section titles and content
        section_texts = []
        for section in sections:
            # Combine title and first 500 chars of content for embedding
            content_preview = section["content"][:500] if section["content"] else ""
            combined_text = f"{section['title']} {content_preview}"
            section_texts.append(combined_text)
        
        if not section_texts:
            return []
        
        section_embeddings = self.model.encode(section_texts)
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        
        # Add keyword-based scoring
        for i, section in enumerate(sections):
            keyword_score = self._calculate_keyword_score(section, persona, job)
            # Combine semantic similarity with keyword matching
            final_score = similarities[i] * 0.7 + keyword_score * 0.3
            section["relevance_score"] = float(final_score)
        
        # Sort by relevance score
        sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(sections):
            section["importance_rank"] = i + 1
        
        return sections
    
    def _calculate_keyword_score(self, section: Dict, persona: str, job: str) -> float:
        """Calculate keyword-based relevance score"""
        text = f"{section['title']} {section['content']}".lower()
        persona_lower = persona.lower()
        job_lower = job.lower()
        
        score = 0.0
        
        # Direct matches with persona and job keywords
        persona_words = persona_lower.split()
        job_words = job_lower.split()
        
        for word in persona_words + job_words:
            if len(word) > 3 and word in text:
                score += 0.1
        
        # Domain-specific keyword matching
        for domain, keywords in self.domain_keywords.items():
            if domain in persona_lower or domain in job_lower:
                for keyword in keywords:
                    if keyword in text:
                        score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_subsections(self, section: Dict) -> List[Dict]:
        """Extract subsections from a main section"""
        content = section["content"]
        subsections = []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Take most relevant paragraphs as subsections
        for i, paragraph in enumerate(paragraphs[:3]):  # Top 3 paragraphs
            if len(paragraph) > 100:  # Minimum length for meaningful subsection
                subsections.append({
                    "document": section["document"],
                    "section_title": section["title"],
                    "refined_text": paragraph,
                    "page_number": section["page"],
                    "subsection_rank": i + 1
                })
        
        return subsections
    
    def process_document_collection(self, input_dir: str, persona: str, job: str) -> Dict[str, Any]:
        """Process entire document collection"""
        
        all_sections = []
        input_documents = []
        
        # Process all PDFs in the input directory
        for pdf_file in Path(input_dir).glob("*.pdf"):
            logger.info(f"Processing {pdf_file.name}")
            input_documents.append(pdf_file.name)
            
            sections = self.extract_sections_from_pdf(str(pdf_file))
            all_sections.extend(sections)
        
        # Calculate relevance scores for all sections
        ranked_sections = self.calculate_relevance_scores(all_sections, persona, job)
        
        # Take top 10 most relevant sections
        top_sections = ranked_sections[:10]
        
        # Extract subsections from top sections
        all_subsections = []
        for section in top_sections[:5]:  # Top 5 sections for subsection analysis
            subsections = self.extract_subsections(section)
            all_subsections.extend(subsections)
        
        # Prepare output
        result = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section["document"],
                    "page_number": section["page"],
                    "section_title": section["title"],
                    "importance_rank": section["importance_rank"]
                }
                for section in top_sections
            ],
            "sub_section_analysis": all_subsections
        }
        
        return result

def main():
    """Main execution function"""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load persona and job from config file or environment
    config_file = Path(input_dir) / "config.json"
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        persona = config.get("persona", "General Researcher")
        job = config.get("job_to_be_done", "Analyze documents for key insights")
    else:
        # Default values for testing
        persona = "PhD Researcher in Computational Biology"
        job = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
    
    # Initialize analyzer
    analyzer = PersonaDocumentAnalyzer()
    
    # Process documents
    try:
        result = analyzer.process_document_collection(input_dir, persona, job)
        
        # Write output
        output_file = Path(output_dir) / "challenge1b_output.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete. Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()