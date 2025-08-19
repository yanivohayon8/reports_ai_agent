from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(**kwargs):
    default_kwargs = {
        "chunk_size":300,
        "chunk_overlap":50
        # TODO: length function of tokens
    }
    default_kwargs.update(kwargs)
    
    return RecursiveCharacterTextSplitter(**default_kwargs)