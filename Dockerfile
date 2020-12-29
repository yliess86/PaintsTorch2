FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgtk2.0-dev

WORKDIR /app

ADD requirements.txt ./
RUN pip install -r requirements.txt

RUN python3 -c "from torchvision.models import vgg16; _ = vgg16(pretrained=True);"
RUN python3 -c "from torchvision.models import inception_v3; _ = inception_v3(pretrained=True);"

ADD paintstorch2 paintstorch2/
ADD experiments experiments/

ENTRYPOINT ["python3", "-m"]