FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libgtk2.0-dev python-opencv

WORKDIR /app

ADD requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN python -c "import cv2"
RUN python -c "from torchvision.models import vgg16; _ = vgg16(pretrained=True);"
RUN python -c "from torchvision.models import inception_v3; _ = inception_v3(pretrained=True);"

ADD paintstorch paintstorch/
ADD models models/

ENTRYPOINT ["python", "-m"]