from agents.router_agent.router import classify_query, generate_response, Classification

def test_classify_query_return_classification():
    query = "Summerize the burglary report."
    result = classify_query(query)
    assert isinstance(result, Classification)
    assert result.type in ["summary", "needle", "tableQA"]
    assert result.complexity in ["simple", "complex"]   

def test_generate_response_summary_return_string():
    query = "Summerize the burglary report."
    classification = Classification(type="summary", complexity="simple", reasoning="Test reasoning")
    context = "Burglary event happened in July 2024 with damages of $5000."
    response = generate_response(query, classification, context)
    assert isinstance(response, str)
    assert len(response) > 0

def test_generate_response_needle_return_string():
    query = "Find the date of the burglary."
    classification = Classification(type="needle", complexity="simple", reasoning="Test reasoning")
    context = "Burglary event happened in July 2024 with damages of $5000."
    response = generate_response(query, classification, context)
    assert isinstance(response, str)
    assert len(response) > 0

def test_generate_response_tableQA_return_string():
    query = "What is the total amount of damages?"
    classification = Classification(type="tableQA", complexity="simple", reasoning="Test reasoning")
    context = "Table 1: Burglary Report\nDate: July 2024\nAmount: $5000"
    response = generate_response(query, classification, context)
    assert isinstance(response, str)
    assert len(response) > 0
    
        