FROM tiangolo/meinheld-gunicorn-flask:python3.7
COPY ./app/requirements.txt /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
COPY ./app /app
