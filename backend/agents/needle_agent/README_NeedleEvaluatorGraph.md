# NeedleEvaluatorGraph - Graph RAG Evaluation System

## Overview

The `NeedleEvaluatorGraph` is a specialized RAGAS evaluation system designed to evaluate the performance of `NeedleAgentGraph` (Graph RAG) implementations. It provides comprehensive evaluation metrics for Graph RAG-based needle retrieval systems.

## Features

- **RAGAS Integration**: Uses RAGAS metrics for evaluation
- **Graph RAG Support**: Specifically designed for Graph RAG agents
- **Comprehensive Metrics**: Faithfulness, Context Recall, and LLM Context Precision
- **Configuration Driven**: YAML-based configuration system
- **CLI Interface**: Command-line evaluation tool
- **Result Export**: JSON output with detailed metrics

## Files

- `needle_evaluator_graph.py` - Main evaluator class
- `needle_evaluator_graph_cli.py` - Command-line interface
- `needle_evaluator_graph_config.yaml` - Configuration file
- `run_evaluation_example.py` - Example usage script
- `test_needle_graph_evaluator.py` - Test script

## Quick Start

### 1. Test Setup
```bash
python backend/agents/needle_agent/test_needle_graph_evaluator.py
```

### 2. Run Evaluation
```bash
python backend/agents/needle_agent/needle_evaluator_graph_cli.py
```

### 3. Custom Configuration
```bash
python backend/agents/needle_agent/needle_evaluator_graph_cli.py \
  --config custom_config.yaml \
  --dataset path/to/dataset.jsonl \
  --output custom_results.json
```

## Configuration

The system uses a YAML configuration file with the following structure:

```yaml
evaluation:
  dataset:
    path: "backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl"
  
  output:
    path: "needle_graph_evaluation_results.json"
  
  metrics:
    selected: []  # Empty means all metrics
  
  llm:
    model: "gpt-4o-mini"
    temperature: 0.0
  
  router:
    config_path: "backend/agents/router_agent/config.yaml"
  
  settings:
    batch_size: 10
    verbose: true
    timeout: 600
  
  graph_rag:
    verify_graph_usage: true
    source_attribution: true
```

## Metrics

### Available Metrics

1. **Faithfulness**: Measures how faithful the generated answer is to the retrieved context
2. **Context Recall**: Measures how well the retrieved context covers the information needed
3. **LLM Context Precision**: Measures the precision of the LLM's use of the retrieved context

### Metric Selection

- **All Metrics**: Leave `selected: []` empty (default)
- **Specific Metrics**: Specify desired metrics:
  ```yaml
  metrics:
    selected: ["faithfulness", "context_recall"]
  ```

## Prerequisites

1. **Graph RAG Setup**: Ensure your router is configured to use `NeedleAgentGraph`
2. **Dataset**: Ground truth dataset in JSONL format
3. **LLM Access**: OpenAI API key configured
4. **Dependencies**: RAGAS, LangChain, and other required packages

## Troubleshooting

### Common Issues

1. **"Not using NeedleAgentGraph"**
   - Check your router configuration
   - Ensure `NeedleAgent.type: graph` and `GraphRAG.enabled: true`

2. **Dataset not found**
   - Verify the dataset path in configuration
   - Check file permissions

3. **LLM initialization failed**
   - Verify OpenAI API key
   - Check internet connectivity

### Debug Mode

Enable verbose output in configuration:
```yaml
settings:
  verbose: true
```

## Example Output

```
🚀 Starting NeedleAgentGraph (Graph RAG) Evaluation...
📊 Dataset: backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl
🎯 Output: needle_graph_evaluation_results.json
🤖 Model: gpt-4o-mini

🔧 Creating router with Graph RAG agents...
✅ Router created with NeedleAgentGraph
🎯 Confirmed: Using NeedleAgentGraph (Graph RAG)

📈 Running evaluation...
📊 Evaluation Results:
Faithfulness: 0.8234
Context Recall: 0.9123
LLM Context Precision: 0.7845

💾 Results saved to needle_graph_evaluation_results.json
🎉 Evaluation completed successfully!
```

## Advanced Usage

### Programmatic Evaluation

```python
from backend.agents.needle_agent.needle_evaluator_graph import NeedleEvaluatorGraph
from backend.core.api_utils import get_llm_langchain_openai

# Initialize
llm = get_llm_langchain_openai(model="gpt-4o-mini")
evaluator = NeedleEvaluatorGraph(dataset_path, llm)

# Run evaluation
result = evaluator.evaluate(agent)

# Save results
evaluator.save_results(result, output_path, agent)
```

### Custom Metrics

```python
# Select specific metrics
evaluator = NeedleEvaluatorGraph(
    dataset_path, 
    llm, 
    selected_metrics=["faithfulness", "context_recall"]
)
```

## Support

For issues or questions:
1. Check the test script output
2. Verify configuration settings
3. Ensure Graph RAG is properly configured
4. Check dataset format and accessibility
