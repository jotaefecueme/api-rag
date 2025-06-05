FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8080

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
    "-b", "0.0.0.0:8080", \
    "--workers", "1", "--threads", "1", \
    "--timeout", "30", "--graceful-timeout", "15", \
    "--keep-alive", "60", \
    "--log-level", "info", \
    "--access-logfile", "-", \
    "--error-logfile", "-"]