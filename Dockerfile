FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y

COPY cloud.py .
COPY yolo11n.pt .

CMD ["python", "cloud.py"]