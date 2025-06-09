FROM python:3.10

WORKDIR /app

COPY . .

# pip, setuptools, wheel 최신 버전으로 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 패키지 설치
RUN pip install -r a.txt

CMD ["python", "server.py"]
