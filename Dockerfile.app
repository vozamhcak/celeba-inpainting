FROM python:3.10-slim

WORKDIR /app

COPY requirements.app.txt /app/requirements.app.txt
RUN pip install --no-cache-dir -r requirements.app.txt

COPY code/deployment/app /app/app

ENV API_URL=http://api:8000

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
