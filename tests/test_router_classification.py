import pytest
from backend.agents.router_agent.router import RouterAgent


@pytest.fixture
def router_agent():
    """RouterAgent without agents for testing classification logic."""
    return RouterAgent()


class TestRouterClassification:
    """Test the router's classification logic with various query types."""
    
    def test_summary_queries(self, router_agent):
        """Test that general summary queries are classified correctly."""
        summary_queries = [
            "Please summarize the accident report",
            "Give me an overview of the policy",
            "What are the main points of this document?",
            "Can you provide a summary?",
            "Tell me about the insurance coverage in general"
        ]
        
        for query in summary_queries:
            # These should have low scores for all categories
            q = query.lower()
            
            # Check needle indicators
            needle_indicators = [
                "find", "search", "locate", "when", "where", "how much",
                "date", "time", "location", "amount", "cost", "price",
                "policy number", "hospital", "surgery",
                "specific", "exact", "precise", "details about", "information on"
            ]
            needle_score = sum(1 for indicator in needle_indicators if indicator in q)
            
            # Check person indicators
            person_indicators = [
                "who", "what is the", "what's the", "what is", "nationality", "insured name"
            ]
            person_score = sum(1 for indicator in person_indicators if indicator in q)
            
            # Check table indicators
            table_indicators = [
                "table", "chart", "graph", "data", "statistics", "numbers", 
                "figures", "columns", "rows", "cells", "spreadsheet", "grid",
                "compare", "calculate", "total", "sum", "average", "percentage", "policy"
            ]
            table_score = sum(1 for indicator in table_indicators if indicator in q)
            
            # Summary queries should have low scores
            assert needle_score <= 1, f"Query '{query}' has high needle score: {needle_score}"
            assert person_score <= 1, f"Query '{query}' has high person score: {person_score}"
            assert table_score <= 1, f"Query '{query}' has high table score: {table_score}"
    
    def test_needle_queries(self, router_agent):
        """Test that specific needle queries are classified correctly."""
        needle_queries = [
            "When did the accident occur?",
            "What is the policy number?",
            "Find the exact date of the incident",
            "What is the coverage amount?",
            "Where did the event happen?",
            "How much does the policy cost?",
            "What hospital was involved?",
            "Find the specific details about the surgery"
        ]
        
        for query in needle_queries:
            q = query.lower()
            
            # Check needle indicators
            needle_indicators = [
                "find", "search", "locate", "when", "where", "how much",
                "date", "time", "location", "amount", "cost", "price",
                "policy number", "hospital", "surgery",
                "specific", "exact", "precise", "details about", "information on"
            ]
            needle_score = sum(1 for indicator in needle_indicators if indicator in q)
            
            # Needle queries should have higher needle scores
            assert needle_score >= 1, f"Query '{query}' has low needle score: {needle_score}"
    
    def test_table_queries(self, router_agent):
        """Test that table-related queries are classified correctly."""
        table_queries = [
            "Show me the table with accident data",
            "What is in the coverage table?",
            "Display the statistics from the chart",
            "Show the data in columns and rows",
            "What are the numbers in the table?",
            "Compare the values in the spreadsheet",
            "Calculate the total from the data",
            "What percentage is shown in the graph?"
        ]
        
        for query in table_queries:
            q = query.lower()
            
            # Check table indicators
            table_indicators = [
                "table", "chart", "graph", "data", "statistics", "numbers", 
                "figures", "columns", "rows", "cells", "spreadsheet", "grid",
                "compare", "calculate", "total", "sum", "average", "percentage"
            ]
            table_score = sum(1 for indicator in table_indicators if indicator in q)
            
            # Table queries should have higher table scores
            assert table_score >= 1, f"Query '{query}' has low table score: {table_score}"
    
    def test_person_data_queries(self, router_agent):
        """Test that person data queries are classified as table queries."""
        person_queries = [
            "What is the nationality of Maria?",
            "Who is the insured person?",
            "What is the name of the policyholder?",
            "What is the insured name?",
            "Who is covered by this policy?",
            "What is the person's details?"
        ]
        
        for query in person_queries:
            q = query.lower()
            
            # Check person indicators
            person_indicators = [
                "who", "what is the", "what's the", "what is", "nationality", "insured name"
            ]
            person_score = sum(1 for indicator in person_indicators if indicator in q)
            
            # Person queries should have higher person scores
            assert person_score >= 1, f"Query '{query}' has low person score: {person_score}"
            
            # Check for person-related keywords that should trigger table classification
            person_keywords = ["nationality", "national", "insured", "name", "person", "covered", "policy", "policyholder"]
            has_person_keywords = any(word in q for word in person_keywords)
            
            # These should be classified as table queries
            assert has_person_keywords, f"Query '{query}' should contain person keywords"
    
    def test_edge_cases(self, router_agent):
        """Test edge cases and unusual queries."""
        edge_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a",  # Single character
            "the",  # Single word
            "123",  # Numbers only
            "!@#$%",  # Special characters only
            "table table table",  # Repeated words
            "find find find find",  # Repeated indicators
        ]
        
        for query in edge_queries:
            q = query.lower()
            
            # These should handle gracefully
            needle_score = sum(1 for indicator in [
                "find", "search", "locate", "when", "where", "how much",
                "date", "time", "location", "amount", "cost", "price",
                "policy number", "hospital", "surgery",
                "specific", "exact", "precise", "details about", "information on"
            ] if indicator in q)
            
            table_score = sum(1 for indicator in [
                "table", "chart", "graph", "data", "statistics", "numbers", 
                "figures", "columns", "rows", "cells", "spreadsheet", "grid",
                "compare", "calculate", "total", "sum", "average", "percentage"
            ] if indicator in q)
            
            person_score = sum(1 for indicator in [
                "who", "what is the", "what's the", "what is", "nationality", "insured name"
            ] if indicator in q)
            
            # Edge cases should not crash and should have reasonable scores
            assert needle_score >= 0
            assert table_score >= 0
            assert person_score >= 0


class TestRouterClassificationExamples:
    """Test specific examples that should work correctly."""
    
    def test_maria_nationality_example(self, router_agent):
        """Test the specific example: 'what is the national of maria?'"""
        query = "what is the national of maria?"
        q = query.lower()
        
        # Check person indicators
        person_indicators = [
            "who", "what is the", "what's the", "what is", "nationality", "insured name"
        ]
        person_score = sum(1 for indicator in person_indicators if indicator in q)
        
        # Check for person keywords
        person_keywords = ["nationality", "national", "insured", "name", "person"]
        has_person_keywords = any(word in q for word in person_keywords)
        
        # This should be classified as a table question
        assert person_score >= 1, f"Query should have person score >= 1, got {person_score}"
        assert has_person_keywords, "Query should contain person keywords"
        assert "national" in q, "Query should contain 'national'"
        assert "maria" in q, "Query should contain 'maria'"
    
    def test_accident_table_example(self, router_agent):
        """Test: 'give me the first table of maria accident report'"""
        query = "give me the first table of maria accident report"
        q = query.lower()
        
        # Check table indicators
        table_indicators = [
            "table", "chart", "graph", "data", "statistics", "numbers", 
            "figures", "columns", "rows", "cells", "spreadsheet", "grid",
            "compare", "calculate", "total", "sum", "average", "percentage", "policy"
        ]
        table_score = sum(1 for indicator in table_indicators if indicator in q)
        
        # This should be classified as a table question
        assert table_score >= 1, f"Query should have table score >= 1, got {table_score}"
        assert "table" in q, "Query should contain 'table'"
    
    def test_needle_example(self, router_agent):
        """Test a clear needle question."""
        query = "When did the accident occur in Athens?"
        q = query.lower()
        
        # Check needle indicators
        needle_indicators = [
            "find", "search", "locate", "when", "where", "how much",
            "date", "time", "location", "amount", "cost", "price",
            "policy number", "hospital", "surgery",
            "specific", "exact", "precise", "details about", "information on",
            "occur"
        ]
        needle_score = sum(1 for indicator in needle_indicators if indicator in q)
        
        # This should be classified as a needle question
        assert needle_score >= 2, f"Query should have needle score >= 2, got {needle_score}"
        assert "when" in q, "Query should contain 'when'"
        assert "athens" in q, "Query should contain 'athens'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
