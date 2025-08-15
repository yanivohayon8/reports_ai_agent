from core.pdf_reader import read_pdf


class SummaryAgent:

    def _read_pdf(self,pdf_path):
        return read_pdf(pdf_path)

    def summarize_single_pdf(self,pdf_path, method:str):
        text = self._read_pdf(pdf_path)
        if method == "map_reduce":
            return self._summarize_map_reduce(text)
        elif method == "iterative":
            return self._summarize_iterative_refinement(text)
        else:
            raise ValueError(f"Invalid summary method: {method}")

    def summarize_multiple_pdfs(self,pdf_paths, method:str):
        raise NotImplementedError("Multiple pdfs summary method not implemented")

    def _summarize_map_reduce(self,text):
        raise NotImplementedError("Map reduce summary method not implemented")
    
    def _summarize_iterative_refinement(self,text):
        raise NotImplementedError("Iterative refinement summary method not implemented")



