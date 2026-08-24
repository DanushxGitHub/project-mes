FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT}"]
