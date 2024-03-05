
# RAG on Open-Source LLM
We build a basic RAG on Open-Source LLMs from huggingface using LangChain.
This repository tests code on a small scraped-website sample.

## Repository Structure
```
└─ src
   ├─ __pycache__
   │  └─ helpers.cpython-312.pyc
   ├─ helpers.py
   └─ main.py 
├─ README.md
├─ Dockerfile
├─ environment.yml
├─ requirements.txt
└─ data/vector_db ##will be created once main.py is executed
```

## Instructions to set-up
* Installation
Users can use any of the following three ways: 
    - Using Docker image
        * sudo docker build -t rag_container .
        * sudo docker run rag_container
        * sudo docker run -it rag_container /bin/bash  ##to run in interactive mode
    - Using conda (recommended)
    ```
    conda env create -f environment.yml
    ```
    - Using pip
    ```
    pip install -r requirements.txt
    ```
* Setting-up HuggingFace Access Token
    - Log in to [HuggingFace.co](https://huggingface.co/)
    - Click on your profile icon at the top-right corner, then choose [“Settings.”](https://huggingface.co/settings/)
    - In the left sidebar, navigate to [“Access Token”](https://huggingface.co/settings/tokens)
    - Generate a new access token, assigning it the “write” role.
* Check src/main.py for running RAG with LangChain.
    ```
    python src/main.py
    ```

## References
* Codebase built upon [Awesome-RAG](https://github.com/lucifertrj/Awesome-RAG).
* [Langchain documentation](https://python.langchain.com/docs/modules/data_connection/)


