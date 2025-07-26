# Round 1B: Persona-Driven Document Intelligence

## Overview
This solution analyzes collections of PDF documents and extracts the most relevant sections based on a specific persona and their job-to-be-done, using advanced NLP techniques and semantic similarity.

## Approach

### 1. Document Processing Pipeline
- **Section Extraction**: Identifies document sections using font analysis and pattern recognition
- **Content Analysis**: Extracts full text content for each section
- **Hierarchical Structure**: Maintains document structure and page references

### 2. Semantic Relevance Scoring
The solution combines multiple approaches for accurate relevance assessment:

#### Semantic Similarity
- Uses SentenceTransformer model (all-MiniLM-L6-v2) for embedding generation
- Computes cosine similarity between persona/job queries and section content
- Lightweight model (80MB) optimized for CPU inference

#### Keyword-Based Scoring
- Domain-specific keyword matching for different professional contexts
- Direct persona and job keyword matching
- Weighted combination with semantic scores (70% semantic, 30% keyword)

### 3. Multi-Domain Intelligence
Pre-configured domain expertise for:
- **Research**: Methodology, results, literature review focus
- **Business**: Revenue, market analysis, strategic insights
- **Technical**: Architecture, implementation, specifications
- **Educational**: Concepts, theories, learning materials
- **Financial**: Budget analysis, investment metrics

### 4. Section Ranking and Subsection Analysis
- Ranks all sections across document collection by relevance
- Extracts top subsections from most relevant sections
- Provides importance rankings for navigation prioritization

## Technical Architecture

### Model Selection
- **SentenceTransformer (all-MiniLM-L6-v2)**:
  - Size: ~80MB (well under 1GB limit)
  - Fast CPU inference
  - High-quality semantic embeddings
  - Multilingual support

### Processing Pipeline
1. **Document Ingestion**: Process all PDFs in collection
2. **Section Extraction**: Identify logical document sections
3. **Content Embedding**: Generate semantic embeddings for all sections
4. **Relevance Scoring**: Compute persona-job relevance scores
5. **Ranking & Selection**: Rank and select top relevant content
6. **Subsection Analysis**: Extract granular subsections from key sections

## Libraries Used
- **PyMuPDF (fitz)**: PDF text extraction and processing
- **sentence-transformers**: Semantic embeddings and similarity
- **scikit-learn**: Cosine similarity calculations
- **numpy**: Numerical computations for embeddings
- **pathlib**: File system operations

## Build and Run Instructions

### Configuration
Create a `config.json` file in the input directory:
```json
{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare comprehensive literature review focusing on methodologies and benchmarks"
}
```

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t persona-doc-analyzer:latest .
```

### Running the Solution
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none persona-doc-analyzer:latest
```

### Expected Output Format
```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Literature review preparation",
    "processing_timestamp": "2025-01-XX"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Methodology",
      "importance_rank": 1
    }
  ],
  "sub_section_analysis": [
    {
      "document": "doc1.pdf",
      "section_title": "Methodology",
      "refined_text": "Detailed methodology content...",
      "page_number": 3,
      "subsection_rank": 1
    }
  ]
}
```

## Performance Characteristics
- **Processing Time**: <60 seconds for 3-5 document collections
- **Model Size**: ~200MB total (including dependencies)
- **Memory Usage**: Optimized for 16GB RAM systems
- **CPU Architecture**: AMD64 optimized
- **Network**: Fully offline operation

## Key Features

### Intelligence Features
- **Context-Aware Analysis**: Understands persona expertise and objectives
- **Cross-Document Insights**: Analyzes relationships across document collection
- **Hierarchical Relevance**: Provides both section and subsection level analysis
- **Domain Adaptability**: Works across research, business, technical, and educational domains

### Technical Features
- **Fast Processing**: Optimized for quick turnaround on document collections
- **Scalable Architecture**: Handles varying document collection sizes
- **Robust Error Handling**: Graceful degradation for problematic documents
- **Memory Efficient**: Streaming processing for large document collections

## Sample Test Cases Supported

### Academic Research Scenario
- **Documents**: Research papers on specific topics
- **Persona**: PhD Researcher, Post-doc, Academic
- **Jobs**: Literature reviews, methodology analysis, benchmark comparisons

### Business Analysis Scenario  
- **Documents**: Annual reports, market research, financial statements
- **Persona**: Investment Analyst, Business Consultant, Strategy Manager
- **Jobs**: Market analysis, competitive intelligence, financial assessment

### Educational Content Scenario
- **Documents**: Textbooks, course materials, reference guides
- **Persona**: Student, Educator, Training Specialist
- **Jobs**: Exam preparation, curriculum development, concept mastery

## Algorithm Details

### Section Detection Algorithm
1. **Font-Based Analysis**: Identifies headers using font size and styling
2. **Pattern Matching**: Recognizes common section header patterns
3. **Structural Analysis**: Maintains document hierarchy and relationships

### Relevance Scoring Formula
```
Final_Score = (Semantic_Similarity × 0.7) + (Keyword_Score × 0.3)
```

Where:
- **Semantic_Similarity**: Cosine similarity between embeddings
- **Keyword_Score**: Weighted keyword matching score

### Ranking Strategy
- Cross-document section comparison
- Importance ranking based on relevance scores
- Subsection extraction from top-ranked sections only

## Error Handling and Edge Cases
- **Malformed PDFs**: Continues processing with available documents
- **Empty Sections**: Filters out sections with insufficient content
- **Missing Persona/Job**: Uses sensible defaults for generic analysis
- **Large Documents**: Implements content truncation for embedding efficiency

## Future Enhancements
- **Multi-language Support**: Enhanced support for non-English documents
- **Custom Domain Models**: Fine-tuned models for specific professional domains
- **Interactive Refinement**: Feedback loops for improving relevance scoring
- **Visual Content Analysis**: Integration of figure and table analysis