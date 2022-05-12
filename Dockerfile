FROM python:3.8.12-buster

COPY api /api
COPY TailsAndWhales /TailsAndWhales
COPY image /image
COPY model_7 /model_7
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
