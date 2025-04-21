# Dockerfile
FROM apache/airflow:2.8.4

# Airflow 사용자로 전환
USER airflow

# Google Cloud SDK 설치
RUN curl -sSL https://sdk.cloud.google.com > /tmp/install_gcloud.sh && \
    bash /tmp/install_gcloud.sh --install-dir=/home/airflow --disable-prompts && \
    rm /tmp/install_gcloud.sh
ENV PATH $PATH:/home/airflow/google-cloud-sdk/bin

# 추가 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.8.txt"