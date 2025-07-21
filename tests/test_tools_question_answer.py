import sys
sys.path.append("./")
from src.tools.question_answer import QuestionAnswer
from src.utils import read_pdf_text
import os
import shutil

def test_question_answer():
    text = read_pdf_text("data/report.pdf")
    question_answer = QuestionAnswer.from_recursive_splitter(text)

    answer  = question_answer.answer_question("What is the main topic of the document?")
    print(answer)


def test_preprocess_and_load_vector_store():
    pdf_path = "data/report.pdf"
    output_folder = "vector_stores/test_store"
    # Clean up before test
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # Preprocess and save vector store
    QuestionAnswer.preprocess_pdf_to_vector_store(pdf_path, output_folder)
    # Load from saved vector store
    qa_loaded = QuestionAnswer.from_faiss_vector_store(output_folder)
    answer = qa_loaded.answer_question("What is the main topic of the document?")
    print(answer)
    # Clean up after test
    shutil.rmtree(output_folder)


