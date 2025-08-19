from indexer.indexer import FAISSIndexer

class NeedleAgent():
    
    def __init__(self, faiss_indexer:FAISSIndexer) -> None:
        self.faiss_indexer = faiss_indexer

    def search(self, query:str):
        return "Implement search of NeedleAgent"