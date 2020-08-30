FROM tiangolo/meinheld-gunicorn-flask:python3.8
COPY ./app/requirements.txt /app
RUN pip3 install --upgrade pip && \
    pip3 install pillow && \
    python -m pip install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install -U jsonpickle
RUN pip3 install matplotlib
COPY ./app /app
