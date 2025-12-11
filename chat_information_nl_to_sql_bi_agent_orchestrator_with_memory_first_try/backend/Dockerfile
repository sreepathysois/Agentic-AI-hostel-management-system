# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps for pyodbc & builds
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    unixodbc-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl gnupg ca-certificates apt-transport-https unixodbc-dev build-essential

# Add Microsoft repo WITHOUT apt-key
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/microsoft-prod.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/22.04/prod jammy main" \
      > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install -y msodbcsql18


# Copy requirements & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

#RUN chmod +x scripts/reindex_kb.sh
RUN chmod +x /app/entrypoint.sh || true
RUN chmod +x /app/scripts/reindex_kb.sh || true

#CMD ["./start.sh"]

# Expose port
#EXPOSE 8000

#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#WORKDIR /app
#COPY ./backend/requirements.txt /app/requirements.txt
#RUN pip install --upgrade pip
#RUN pip install -r /app/requirements.txt

#COPY backend /app

#RUN chmod +x /app/entrypoint.sh || true
#RUN chmod +x /app/scripts/reindex_kb.sh || true

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
