
from langchain.chains import RetrievalQA, LLMChain
from helpers import load_document_from_url, get_prompt, load_model, create_vector_db, load_from_db
import argparse
import os
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
import sys
sys.path.append("../") 

def parse_arguments():
    parser = argparse.ArgumentParser(description='RAG Pipeline using open source huggingface models')
    parser.add_argument('--url', type=str, help='URL to scrape for documents', default="https://avani17101.github.io/")
    parser.add_argument('--query', type=str, help='query', default="tell me about Avani")
    parser.add_argument('--k', type=int, help='#sentences to be retrived', default=2)
    parser.add_argument('--model_name', type=str, help='HF model name', default="huggingfaceh4/zephyr-7b-alpha")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    with open("data/auth.txt", 'r') as f:
        HF_TOKEN = f.read()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    
    ## Embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )

    save_path = "data/vector_db/url"+args.url.replace('https://','').replace('/','_')
    if not os.path.exists(save_path):
        vectorstore = create_vector_db(args.url, save_path=save_path, embeddings=embeddings)
    else:
        vectorstore = load_from_db(save_path, embeddings=embeddings)

    ## Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr", #similarity
        search_kwargs={'k': args.k}
    )

    # print("retrived documents \n",retriever.get_relevant_documents(args.query))

    ## Load Open Source Large Language Model
    llm = load_model(args.model_name, {"temperature": 0.5, "max_length": 64, "max_new_tokens":512})

    ## Prompt Template and User Input (Augment - Step 2)
    prompt_template = get_prompt(args.query)

    ## Creating a chain: RAG RetrievalQA chain 
    ## This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a respoinse.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever)

    response = qa.run(prompt_template)
    print("Response on RAG RetrievalQA chain \n \n \n",response)
