
# RAG on Open-Source LLM
We build a basic RAG on Open-Source LLMs from huggingface using LangChain.
This repository tests code on a small scraped-website sample.

## Instructions to set-up
* Installation
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
* Check main.py for running RAG with LangChain.

## References
* Codebase built upon [Awesome-RAG](https://github.com/lucifertrj/Awesome-RAG).
* [Langchain documentation](https://python.langchain.com/docs/modules/data_connection/)


