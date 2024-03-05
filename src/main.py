# Build RAG pipeline using Open Source Large Languages
# In the notebook we will build a Chat with Website use cases using Zephyr 7B model
"""## Import RAG components required to build pipeline"""

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, LLMChain
from helpers import scrape_url, get_prompt, load_model
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='RAG Pipeline using open source huggingface models')
    parser.add_argument('--url', type=str, help='URL to scrape for documents', default="https://avani17101.github.io/")
    parser.add_argument('--query', type=str, help='query', default="Where does Avani work")
    parser.add_argument('--k', type=int, help='#sentences to be retrived', default=4)
    parser.add_argument('--model', type=str, help='HF model name', default="huggingfaceh4/zephyr-7b-alpha")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    with open("token.txt", 'r') as f:
        HF_TOKEN = f.read()
    
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

    # External data/document - ETL
    docs = scrape_url("https://avani17101.github.io/")

    # Text Splitting - Chunking"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20) ##adjust chunk overlap as needed
    chunks = text_splitter.split_documents(docs)
    print(len(chunks))


    ## Embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )

    ## Vector Store - FAISS or ChromaDB
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # search = vectorstore.similarity_search(query)
    #search[0].page_content

    ## Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr", #similarity
        search_kwargs={'k': args.k}
    )

    print("retrived documents",retriever.get_relevant_documents(args.query))

    ## Large Language Model - Open Source
    llm = load_model(args.model, {"temperature": 0.5, "max_length": 64,"max_new_tokens":512})

    ## Prompt Template and User Input (Augment - Step 2)
    prompt = get_prompt(args.query)

    ## RAG RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

    response = qa.run(prompt)
    print("response on RAG RetrievalQA chain",response)


    # """## Chain"""

    # from langchain.schema.runnable import RunnablePassthrough
    # from langchain.schema.output_parser import StrOutputParser
    # from langchain.prompts import ChatPromptTemplate

    # template = f"""
    #  <|system|>
    # You are an AI assistant that follows instruction extremely well.
    # Please be truthful and give direct answers
    # </s>
    #  <|user|>
    #  {query}
    #  </s>
    #  <|assistant|>
    # """

    # prompt = ChatPromptTemplate.from_template(template)

    # rag_chain = (
    #     {"context": retriever,  "query": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # query

    # response = rag_chain.invoke(query)

    # print(response)