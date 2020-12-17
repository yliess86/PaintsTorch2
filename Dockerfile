FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ADD requirements.txt ./
ADD paintstorch2 paintstorch2/

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "-m", "paintstorch2.train"]