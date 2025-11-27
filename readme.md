# Hallucination Detection Framework

Detect potential hallucinations in LLM responses by analyzing consistency across multiple generations.

## Intuition Behind the Approach

The core idea is simple: **factual information should be consistent, while hallucinations tend to be inconsistent.**

When you ask the same question multiple times to an LLM:

- **Factual statements** will be similar across responses
- **Hallucinated content** will vary significantly between responses
- By comparing semantic similarity, we can identify inconsistent statements

### How It Works

1. **Multiple Generations** → Ask the same question 3 times with slight temperature variations
2. **Sentence Comparison** → Break down responses into individual sentences
3. **Semantic Analysis** → Use Sentence-BERT to measure similarity between sentences
4. **Consistency Scoring** → Flag sentences with low similarity scores as potential hallucinations

**Example**: If one response says "The capital of France is Paris" and another says "The capital of France is Lyon" - the system detects this inconsistency.

## Quick Setup

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy sentence-transformers spacy ollama
python -m spacy download en_core_web_sm
```

### 2. Setup Ollama
```bash
ollama pull llama3.2:1b
ollama list  # Should show llama3.2:1b
```

### 3. Run the Application
```bash
python main.py
```

## Usage

1. Run `python main.py`
2. Type your question when prompted
3. System generates 3 responses and analyzes them
4. View results in terminal and CSV file

**Test Questions**:
- "What were the exact words of Abraham Lincoln's secret second inaugural address?" (likely hallucinations)
- "Describe the undiscovered element 119 and its properties." (likely hallucinations) 
- "What is the capital of France?" (consistent facts)

## Output

Creates `hallucination_analysis_TIMESTAMP.csv` with:
- All 3 generated responses
- Detected hallucinations with confidence scores
- Easy-to-read format for analysis