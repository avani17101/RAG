FROM ubuntu:latest

RUN apt update
RUN apt install python3-pip -y

COPY . .
RUN pip install -r requirements.txt
CMD [ "python3", "src/main.py"]