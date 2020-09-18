FROM tiangolo/meinheld-gunicorn-flask:python3.8
COPY ./app/requirements.txt /app
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
RUN sed -i '/errorlog = "-"/a timeout = 60' /gunicorn_conf.py && \
 sed -i '/timeout = 60/a accesslog = "-"' /gunicorn_conf.py
COPY ./app /app
VOLUME /files/pictures


