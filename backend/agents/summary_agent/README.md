# Summary Agent

A powerful and flexible document summarization agent that supports multiple summarization methods and can handle both PDF and text documents.

## Features

- **Multiple Summarization Methods**:
  - **Map-Reduce**: Efficient for large documents, processes chunks in parallel
  - **Iterative Refinement**: Builds summary progressively, good for maintaining context
  - **Query-Based**: Focuses on specific aspects or questions

- **Document Support**:
  - PDF files (using llama-parse)
  - Text files
  - Direct text input

- **Advanced Features**:
  - Configurable text chunking
  - Error handling and recovery
  - Multiple LLM model support
  - Query-focused summarization
  - Output formatting options

## Installation

Ensure you have the required dependencies:

```bash
pip install -r ../../requirements.txt
```

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Summarize a PDF file
python summary_cli.py document.pdf

# Summarize with specific method
python summary_cli.py document.pdf --method iterative

# Use different model
python summary_cli.py document.pdf --method map_reduce --model gpt-4
```

#### Query-Based Summarization

```bash
# Focus on specific aspects
python summary_cli.py document.pdf --method query_based --query "What are the main findings?"

# Ask about specific topics
python summary_cli.py document.pdf --method query_based --query "What does the document say about costs?"
```

#### Advanced Options

```bash
# Custom chunk size and overlap
python summary_cli.py document.pdf --chunk-size 1500 --chunk-overlap 300

# Save output to file
python summary_cli.py document.pdf --output summary.txt

# Verbose output
python summary_cli.py document.pdf --verbose
```

### Programmatic Usage

```python
from agents.summary_agent.summary import SummaryAgent
from core.api_utils import get_llm_langchain_openai
from core.text_splitter import get_text_splitter

# Initialize components
llm = get_llm_langchain_openai(model="gpt-4o-mini")
text_splitter = get_text_splitter(chunk_size=1000, chunk_overlap=200)
summary_agent = SummaryAgent(text_splitter=text_splitter, llm=llm)

# Summarize text
text = "Your document text here..."
result = summary_agent.summarize(text, method="map_reduce")
print(result["answer"])

# Query-based summarization
result = summary_agent.summarize_with_query(
    "What are the key points?", 
    text
)
print(result["answer"])
```

## Summarization Methods

### 1. Map-Reduce (`map_reduce`)

**Best for**: Large documents, parallel processing

**How it works**:
1. Splits document into chunks
2. Summarizes each chunk independently (Map phase)
3. Combines all summaries into final summary (Reduce phase)

**Pros**: Fast, handles large documents well
**Cons**: May lose some context between chunks

### 2. Iterative Refinement (`iterative`)

**Best for**: Maintaining context, progressive understanding

**How it works**:
1. Starts with first chunk
2. Iteratively refines summary with each new chunk
3. Builds comprehensive understanding progressively

**Pros**: Maintains context, good for complex documents
**Cons**: Slower, sequential processing

### 3. Query-Based (`query_based`)

**Best for**: Focused summaries, specific questions

**How it works**:
1. Takes a specific query/question
2. Focuses summarization on relevant information
3. Creates targeted summary addressing the query

**Pros**: Highly focused, answers specific questions
**Cons**: Requires good query formulation

## Configuration

The summary agent can be configured via `config.yaml`:

```yaml
# LLM Configuration
llm:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 2000

# Text Splitting
text_splitter:
  chunk_size: 1000
  chunk_overlap: 200

# Methods
methods:
  map_reduce:
    enabled: true
    max_chunks_per_batch: 10
```

## Error Handling

The summary agent includes robust error handling:

- **Chunk Processing Errors**: Continues with remaining chunks
- **LLM Errors**: Retries with fallback methods
- **File Reading Errors**: Provides clear error messages
- **Configuration Errors**: Uses sensible defaults

## Integration with Router

The summary agent integrates seamlessly with the router system:

```python
# Router automatically routes general queries to summary agent
router = RouterAgent(...)
result = await router.handle("Summarize this document")
# Automatically uses SummaryAgent
```

## Examples

### Example 1: Research Paper Summary

```bash
python summary_cli.py research_paper.pdf --method map_reduce --query "What are the main conclusions?"
```

### Example 2: Business Report Analysis

```bash
python summary_cli.py business_report.pdf --method iterative --output analysis.txt
```

### Example 3: Technical Documentation

```bash
python summary_cli.py technical_doc.pdf --method query_based --query "What are the key features and requirements?"
```

## Troubleshooting

### Common Issues

1. **"LLM not available"**: Check your OpenAI API key and model configuration
2. **"Error reading PDF"**: Ensure llama-parse is properly configured
3. **"No text provided"**: Check if the input file is empty or corrupted

### Performance Tips

- Use `map_reduce` for large documents (>10 pages)
- Use `iterative` for complex documents requiring context
- Use `query_based` for focused analysis
- Adjust chunk size based on document complexity

## Contributing

To extend the summary agent:

1. Add new summarization methods to `summary.py`
2. Create corresponding prompts in `summary_prompts.py`
3. Update the CLI to support new methods
4. Add tests for new functionality

## License

This summary agent is part of the reports_ai_agent project.

