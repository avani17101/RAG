
from langchain.document_loaders import WebBaseLoader
import nest_asyncio
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

def load_document_from_url(url):
    '''
    scrapes a given URL and returns 
    '''
    nest_asyncio.apply()
    loader = WebBaseLoader(url)
    loader.requests_per_second = 1
    docs = loader.aload()
    return docs

def create_vector_db(url,save_path, embeddings):
    ## Load Document(s) 
    ## can use any document loader https://python.langchain.com/docs/modules/data_connection/document_loaders/
    docs = load_document_from_url(url)

    ## Text Splitting - Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20) ##adjust chunk overlap as needed
    chunks = text_splitter.split_documents(docs)

    
    ## Vector Store
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=save_path)

    # search = vectorstore.similarity_search(query)
    #search[0].page_content
    return vectorstore

def load_from_db(path, embeddings):
    return Chroma(persist_directory=path, embedding_function=embeddings)

        

def load_model(model_name, kwargs):
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs=kwargs
    )
    return llm

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
