FROM tiangolo/uvicorn-gunicorn:python3.8


COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app