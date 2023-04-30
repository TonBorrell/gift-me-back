FROM python:3.9
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
COPY ./db /code/db
COPY ./functions /code/functions
COPY ./config /code/config
CMD [ "uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8081" ]
