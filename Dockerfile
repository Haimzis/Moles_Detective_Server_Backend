FROM tiangolo/meinheld-gunicorn-flask:python3.8
COPY ./app/requirements.txt /app
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
COPY ./app /app
