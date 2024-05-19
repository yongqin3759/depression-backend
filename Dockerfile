FROM tiangolo/uvicorn-gunicorn:python3.8


COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app

WORKDIR /app/app


# Expose port 9000
EXPOSE 9000

# Command to run the application
CMD ["python3", "main.py"]