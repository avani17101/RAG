
from langchain.document_loaders import WebBaseLoader
import nest_asyncio
from langchain.llms import HuggingFaceHub

def scrape_url(url):
    '''
    scrapes a given URL and returns 
    '''
    nest_asyncio.apply()
    loader = WebBaseLoader(url)
    loader.requests_per_second = 1
    docs = loader.aload()
    return docs

def load_model(model_name, kwargs):
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs=kwargs
    )

def get_prompt(query):
    prompt = f"""
    <|system|>
    You are an AI assistant that follows instruction extremely well.
    Please be truthful and give direct answers
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    return prompt
