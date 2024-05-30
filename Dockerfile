FROM tiangolo/uvicorn-gunicorn:python3.9

WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ./app/ /code/app/

# specify default commands
CMD ["fastapi", "run", "app/main.py", "--port", "80"]