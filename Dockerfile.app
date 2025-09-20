FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "streamlit==1.25.0" \
    "streamlit-drawable-canvas==0.9.3" \
    requests \
    Pillow

COPY code/deployment/app /app/app

ENV API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
