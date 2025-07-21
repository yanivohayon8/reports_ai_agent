import sys
sys.path.append("./")
from src.tools.question_answer import QuestionAnswer
from src.utils import read_pdf_text

def test_question_answer():
    text = read_pdf_text("data/report.pdf")
    question_answer = QuestionAnswer.from_recursive_splitter(text)

    answer  = question_answer.answer_question("What is the main topic of the document?")

    print(answer)


