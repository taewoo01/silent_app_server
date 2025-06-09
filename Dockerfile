# Python 3.9.0을 사용
FROM python:3.10.0

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트의 모든 파일 복사
COPY . .

# pip 업그레이드 후 필요한 패키지 설치
RUN pip install -r a.txt

# 앱 실행 (main.py 실행하는 경우)
CMD ["python", "server.py"]
