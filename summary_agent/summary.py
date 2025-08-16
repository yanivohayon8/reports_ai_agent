import pathlib
from core.pdf_reader import read_pdf


class SummaryAgent:

    def summarize_single_pdf(self,pdf_path:pathlib.Path, method:str):
        text = read_pdf(pdf_path,format="text")
        return self.summarize(text,method)

    def summarize(self,text:str,method:str):
        if method == "map_reduce":
            return self._summarize_map_reduce(text)
        elif method == "iterative":
            return self._summarize_iterative_refinement(text)
        else:
            raise ValueError(f"Invalid summary method: {method}")

    def _summarize_map_reduce(self,text:str):
        raise NotImplementedError("Map reduce summary method not implemented")
    
    def _summarize_iterative_refinement(self,text:str):
        raise NotImplementedError("Iterative refinement summary method not implemented")



