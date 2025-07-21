from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_openai_model

class QuestionAnswer:

    def __init__(self,text:str):
        self.chat_model = get_openai_model()

        self.transformed_text = self._transform_to_qa(text)


    def _transform_to_qa(self,text:str,num_questions:int=10)->str:
        # TODO: optimize this
        system_message= """You are an expert of writing questions given a text. 
                            Given the following text, write {num_questions} questions and their answers.
                            Use the following format:
                            Question: [question]
                            Answer: [answer]
                            \\n\\ncontext: {context}
                            """
        self.qa_prompt_template = ChatPromptTemplate.from_messages([
            ("system",system_message)
        ])

        qa_promt = self.qa_prompt_template.invoke({"context":text,"num_questions":num_questions})

        self.transformed_text = self.chat_model.invoke(qa_promt)

        return self.transformed_text