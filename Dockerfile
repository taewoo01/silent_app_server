FROM python:3.9-slim

WORKDIR /app

COPY a.txt ./
RUN pip install --no-cache-dir -r a.txt

COPY src/ ./src/

CMD ["python", "src/server.py"]
