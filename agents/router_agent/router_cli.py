import argparse
from llama_index.core import Document
from agents.router_agent.router import classify_query, generate_response
from retrieval.retrieval import HybridRetriever

def main():
    parser = argparse.ArgumentParser(description="Router Agent CLI")
    parser.add_argument("query", type=str, help="Insurance query to process")
    args = parser.parse_args()

    # Example docs
    docs = [
        Document(text="Burglary event on 12/07/2024. Damage: $5000", metadata={"IncidentType": "Burglary"}),
        Document(text="Policy clause: compensation limits", metadata={"SectionType": "Policy"})
    ]
    retriever = HybridRetriever(docs)

    classification = classify_query(args.query)
    nodes = retriever.retrieve(args.query)
    context = "\n\n".join([n.text for n in nodes])
    response = generate_response(args.query, classification, context)

    print("=== Classification ===")
    print(classification.model_dump())
    print("\n=== Retrieved Context ===")
    for n in nodes:
        print("-", n.text)
    print("\n=== Response ===")
    print(response)

if __name__ == "__main__":
    main()
