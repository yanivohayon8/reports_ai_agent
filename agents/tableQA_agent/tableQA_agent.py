from langchain_core.language_models.chat_models import BaseChatModel
from indexer.indexer import FAISSIndexer
from langchain_core.documents import Document
from typing import List
import pandas as pd
from agents.tableQA_agent.tableQA_chunker import TableChunkType

class TableQAgent:
    def __init__(self, faiss_indexer: FAISSIndexer, llm: BaseChatModel):
        self.faiss_indexer = faiss_indexer
        self.llm = llm

    def answer(self, query: str) -> str:
        context,chunks = self._retrieve_context(query)
        answer = self._generate(context, query)
        # TODO: 
        return 

    def _retrieve_context(self, query: str) -> str:
        chunks = self.faiss_indexer.retrieve(query)
        df_metadata = self._get_metadata_stats(chunks)

        filtered_chunks = self._retain_only_chunks_with_both_types_retrieved(df_metadata,chunks)
        arranged_chunks = self._arrange_chunks_by_table(filtered_chunks)
        context = self._concat_chunks(arranged_chunks)

        return context,chunks

    def _get_metadata_stats(self,chunks:List[Document])->pd.DataFrame:
        metadatas = [chunk.metadata for chunk in chunks]
        df_metadata = pd.DataFrame(metadatas)
        df_metadata = df_metadata.reset_index().rename(columns={'index': 'orig_idx'})

        return df_metadata
    
    
    def _retain_only_chunks_with_both_types_retrieved(self,df_metadata:pd.DataFrame,chunks:List[Document])->List[Document]:
        # Keep only groups that contain both "raw" and "description"
        # Find groups with both raw and description
        mask = df_metadata.groupby(['table_index', 'source'])['table_chunk_type'] \
                .transform(lambda x: set(['raw', 'description']).issubset(set(x)))

        # Get original indices
        valid_indices = df_metadata[mask]['orig_idx'].tolist()

        # Filter chunks based on valid indices
        filtered_chunks = [chunks[i] for i in valid_indices]

        return filtered_chunks
    
    def _arrange_chunks_by_table(self,chunks:List[Document])->List[Document]:
        description_chunks = [chunk for chunk in chunks if chunk.metadata["table_chunk_type"] == TableChunkType.DESCRIPTION]
        raw_chunks = [chunk for chunk in chunks if chunk.metadata["table_chunk_type"] == TableChunkType.RAW]

        arranged_chunks = []

        for descri_chunk in description_chunks:
            arranged_chunks.append(descri_chunk)

            for raw_chunk in raw_chunks:
                if raw_chunk.metadata["table_index"] == descri_chunk.metadata["table_index"] and raw_chunk.metadata["source"] == descri_chunk.metadata["source"]:
                    arranged_chunks.append(raw_chunk)
                    break

        return arranged_chunks


    def _concat_chunks(self,chunks:List[Document])->str:
        return "\n\n".join([chunk.page_content for chunk in chunks])
    
    def _generate(self, context: str, query: str) -> str:
        pass

    def _get_chunks_info(self,chunks:List[Document])->tuple[List[str],List[dict]]:
        pass

    def get_used_input(self) -> dict:
        pass

