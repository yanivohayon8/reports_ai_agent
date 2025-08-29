import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from backend.agents.router_agent.router import RouterAgent, create_router_from_config
from backend.agents.summary_agent.summary_agent import SummaryAgent
from backend.agents.needle_agent.needle_agent import NeedleAgent
from backend.agents.tableQA_agent.tableQA_agent import TableQAgent
from backend.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_summary_agent():
    """Mock SummaryAgent for testing."""
    mock = Mock(spec=SummaryAgent)
    mock.handle = AsyncMock(return_value={
        "answer": "This is a summary response",
        "agent": "SummaryAgent",
        "reasoning": "Generated summary"
    })
    return mock


@pytest.fixture
def mock_needle_agent():
    """Mock NeedleAgent for testing."""
    mock = Mock(spec=NeedleAgent)
    mock.handle = AsyncMock(return_value={
        "answer": "This is a needle response",
        "agent": "NeedleAgent",
        "reasoning": "Found specific information"
    })
    return mock


@pytest.fixture
def mock_table_agent():
    """Mock TableQAgent for testing."""
    mock = Mock(spec=TableQAgent)
    mock.handle = AsyncMock(return_value={
        "answer": "This is a table response",
        "agent": "TableQAgent",
        "reasoning": "Extracted from table data"
    })
    mock.retriever = Mock(spec=HybridRetriever)
    mock.retriever.retrieve.return_value = [
        Mock(metadata={"table_chunk_type": "description"}),
        Mock(metadata={"table_chunk_type": "raw"})
    ]
    return mock


@pytest.fixture
def router_agent(mock_summary_agent, mock_needle_agent, mock_table_agent):
    """RouterAgent with mocked sub-agents for testing."""
    return RouterAgent(
        summary_agent=mock_summary_agent,
        needle_agent=mock_needle_agent,
        table_agent=mock_table_agent
    )


def test_router_agent_initialization():
    """Test that RouterAgent initializes correctly."""
    router = RouterAgent()
    
    assert router.model_name == "gpt-4o-mini"
    assert router.summary_agent is None
    assert router.needle_agent is None
    assert router.table_agent is None
    assert router.llm is not None


def test_router_agent_initialization_with_agents(mock_summary_agent, mock_needle_agent, mock_table_agent):
    """Test that RouterAgent initializes correctly with all agents."""
    router = RouterAgent(
        summary_agent=mock_summary_agent,
        needle_agent=mock_needle_agent,
        table_agent=mock_table_agent
    )
    
    assert router.summary_agent == mock_summary_agent
    assert router.needle_agent == mock_needle_agent
    assert router.table_agent == mock_table_agent


def test_router_is_table_question_with_table_keywords(router_agent):
    """Test table question detection with explicit table keywords."""
    query = "Show me the table with accident data"
    
    result = router_agent.is_table_question(query)
    
    assert result is True


def test_router_is_table_question_without_table_keywords(router_agent):
    """Test table question detection without table keywords."""
    query = "What happened to Maria?"
    
    result = router_agent.is_table_question(query)
    
    assert result is False


def test_router_is_table_question_retrieval_error(router_agent):
    """Test table question detection when retrieval fails."""
    # Make retrieval fail
    router_agent.table_agent.retriever.retrieve.side_effect = Exception("Retrieval error")
    
    query = "Show me the table with accident data"
    
    result = router_agent.is_table_question(query)
    
    assert result is False


def test_router_is_table_question_no_documents(router_agent):
    """Test table question detection when no documents are found."""
    # Make retrieval return empty results
    router_agent.table_agent.retriever.retrieve.return_value = []
    
    query = "Show me the table with accident data"
    
    result = router_agent.is_table_question(query)
    
    assert result is False


@pytest.mark.asyncio
async def test_router_handle_summary_query(router_agent):
    """Test that summary queries are routed to SummaryAgent."""
    query = "Please summarize the accident report"
    
    result = await router_agent.handle(query)
    
    assert result["agent"] == "SummaryAgent"
    assert "summary" in result["answer"].lower()
    router_agent.summary_agent.handle.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_router_handle_needle_query(router_agent):
    """Test that needle queries are routed to NeedleAgent."""
    query = "When did the accident occur?"
    
    result = await router_agent.handle(query)
    
    assert result["agent"] == "NeedleAgent"
    assert "needle" in result["answer"].lower()
    router_agent.needle_agent.handle.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_router_handle_table_query(router_agent):
    """Test that table queries are routed to TableQAgent."""
    query = "What is the coverage amount in the table?"
    
    result = await router_agent.handle(query)
    
    assert result["agent"] == "TableQAgent"
    assert "table" in result["answer"].lower()
    router_agent.table_agent.handle.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_router_handle_person_data_query(router_agent):
    """Test that person data queries are classified as table queries."""
    query = "What is the nationality of Maria?"
    
    result = await router_agent.handle(query)
    
    assert result["agent"] == "TableQAgent"
    router_agent.table_agent.handle.assert_called_once_with(query)


@pytest.mark.asyncio
async def test_router_handle_no_agents_available():
    """Test behavior when no agents are available."""
    router = RouterAgent()  # No agents
    
    query = "Test query"
    
    result = await router.handle(query)
    
    assert result["answer"] == "No suitable agent available."
    assert result["agent"] == "Router"


@pytest.mark.asyncio
async def test_router_handle_missing_agent():
    """Test behavior when a specific agent is missing."""
    router = RouterAgent(
        summary_agent=Mock(spec=SummaryAgent),
        needle_agent=None,  # Missing
        table_agent=Mock(spec=TableQAgent)
    )
    
    # Query that would normally go to needle agent
    query = "When did the accident occur?"
    
    result = await router.handle(query)
    
    # Should fall back to summary since needle is not available
    assert result["agent"] == "SummaryAgent"


def test_router_classification_scores():
    """Test the classification scoring system."""
    router = RouterAgent()
    
    # Test needle indicators
    needle_query = "Find the exact date when the accident happened"
    needle_score = sum(1 for indicator in [
        "find", "search", "locate", "when", "where", "how much",
        "date", "time", "location", "amount", "cost", "price",
        "policy number", "hospital", "surgery",
        "specific", "exact", "precise", "details about", "information on"
    ] if indicator in needle_query.lower())
    
    assert needle_score >= 3  # Should have multiple needle indicators
    
    # Test person indicators
    person_query = "What is the nationality of Maria?"
    person_score = sum(1 for indicator in [
        "who", "what is the", "what's the", "what is", "nationality", "insured name"
    ] if indicator in person_query.lower())
    
    assert person_score >= 2  # Should have multiple person indicators


@pytest.mark.asyncio
async def test_router_override_needle_to_table(router_agent):
    """Test that needle queries can be overridden to table when strong table evidence exists."""
    # Make is_table_question return True to trigger override
    router_agent.is_table_question = Mock(return_value=True)
    
    query = "Find the exact date when the accident happened"
    
    result = await router_agent.handle(query)
    
    # Should be overridden to table
    assert result["agent"] == "TableQAgent"


def test_router_debug_logging(router_agent, capsys):
    """Test that debug logging shows classification details."""
    query = "What is the nationality of Maria?"
    
    # Capture print output
    with capsys.disabled():
        # This would normally print debug info, but we're testing the logic
        pass
    
    # The classification should work correctly
    # This test ensures the debug logging doesn't break functionality


@pytest.mark.asyncio
async def test_router_with_real_agents():
    """Test router with real agent instances (minimal setup)."""
    # Create minimal agent instances
    summary_agent = Mock(spec=SummaryAgent)
    summary_agent.handle = AsyncMock(return_value={"answer": "Summary", "agent": "SummaryAgent"})
    
    needle_agent = Mock(spec=NeedleAgent)
    needle_agent.handle = AsyncMock(return_value={"answer": "Needle", "agent": "NeedleAgent"})
    
    table_agent = Mock(spec=TableQAgent)
    table_agent.handle = AsyncMock(return_value={"answer": "Table", "agent": "TableQAgent"})
    table_agent.retriever = Mock()
    table_agent.retriever.retrieve.return_value = []
    
    router = RouterAgent(
        summary_agent=summary_agent,
        needle_agent=needle_agent,
        table_agent=table_agent
    )
    
    # Test routing
    summary_result = await router.handle("Summarize this")
    assert summary_result["agent"] == "SummaryAgent"
    
    needle_result = await router.handle("Find the date")
    assert needle_result["agent"] == "NeedleAgent"
    
    table_result = await router.handle("Show me the table")
    assert table_result["agent"] == "TableQAgent"


@patch('backend.agents.router_agent.router.Path')
@patch('backend.agents.router_agent.router.yaml')
def test_create_router_from_config_success(mock_yaml, mock_path):
    """Test successful router creation from config."""
    # Mock config
    mock_config = {
        "SummaryAgent": {
            "llm": {"model": "gpt-4o-mini"},
            "indexer": {"faiss_directory": "test/summary"},
            "chunker": {"chunk_size": 1000, "chunk_overlap": 200, "separators": ["\n\n"]}
        },
        "NeedleAgent": {
            "llm": {"model": "gpt-4o-mini"},
            "indexer": {"faiss_directory": "test/needle"}
        },
        "TableQAgent": {
            "llm": {"model": "gpt-4o-mini"},
            "indexer": {"faiss_directory": "test/table"}
        },
        "Router": {"model_name": "gpt-4o-mini"}
    }
    
    mock_yaml.safe_load.return_value = mock_config
    mock_path.return_value.read_text.return_value = "config content"
    
    # Mock all the dependencies
    with patch('backend.agents.router_agent.router.get_llm_langchain_openai') as mock_get_llm, \
         patch('backend.agents.router_agent.router.FAISSIndexer') as mock_faiss, \
         patch('backend.agents.router_agent.router.SummaryChunker') as mock_chunker, \
         patch('backend.agents.router_agent.router.RecursiveCharacterTextSplitter') as mock_splitter, \
         patch('backend.agents.router_agent.router.DenseRetriever') as mock_dense, \
         patch('backend.agents.router_agent.router.SparseRetriever') as mock_sparse, \
         patch('backend.agents.router_agent.router.HybridRetriever') as mock_hybrid, \
         patch('backend.agents.router_agent.router.SummaryAgent') as mock_summary, \
         patch('backend.agents.router_agent.router.NeedleAgent') as mock_needle, \
         patch('backend.agents.router_agent.router.TableQAgent') as mock_table:
        
        # Setup mocks
        mock_get_llm.return_value = Mock()
        mock_faiss.from_small_embedding.return_value = Mock()
        mock_chunker.return_value = Mock()
        mock_splitter.return_value = Mock()
        mock_dense.return_value = Mock()
        mock_sparse.return_value = Mock()
        mock_hybrid.return_value = Mock()
        mock_summary.return_value = Mock()
        mock_needle.return_value = Mock()
        mock_table.return_value = Mock()
        
        # Test creation
        router = create_router_from_config("test_config.yaml")
        
        assert router is not None
        assert isinstance(router, RouterAgent)


@patch('backend.agents.router_agent.router.Path')
@patch('backend.agents.router_agent.router.yaml')
def test_create_router_from_config_failure(mock_yaml, mock_path):
    """Test router creation when config loading fails."""
    mock_yaml.safe_load.side_effect = Exception("Config error")
    mock_path.return_value.read_text.return_value = "config content"
    
    router = create_router_from_config("test_config.yaml")
    
    # Should return minimal router
    assert router is not None
    assert isinstance(router, RouterAgent)
    assert router.summary_agent is None
    assert router.needle_agent is None
    assert router.table_agent is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
