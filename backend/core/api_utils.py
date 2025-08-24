import os
import getpass
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

def _update_environment_variable(name,val):
    os.environ[name] = val

def _get_environment_variable(name):
    return os.environ.get(name)

def _verify_environment_variable(name,message=None):
    if not _get_environment_variable(name):
        if message is None:
            message = f"Enter value for {name}:"

        _update_environment_variable(name,getpass.getpass(message))


def verify_openai_api_key():
    _verify_environment_variable("OPENAI_API_KEY")


def get_openai_embeddings(model:str,**kwargs):
    verify_openai_api_key()
    return OpenAIEmbeddings(model=model,**kwargs)


def get_llm_langchain_openai( **chat_settings):
    verify_openai_api_key()

    return ChatOpenAI(**chat_settings)


def verify_and_get_environment_variable(name):
    _verify_environment_variable(name)

    return _get_environment_variable(name)

def verify_llama_parse_api_key():
    _verify_environment_variable("LLAMA_CLOUD_API_KEY")