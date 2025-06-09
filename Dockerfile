FROM python:3.9-slim

WORKDIR /app

COPY a.txt .
RUN pip install -r a.txt

COPY . .

CMD ["python", "src/server.py"]
