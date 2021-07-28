FROM continuumio/miniconda3:latest

WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY webApp /app/
WORKDIR /app



ENTRYPOINT [ "gunicorn" ]
CMD [ "app:app", "--reload", "-w 2", "-b 0.0.0.0:8000" ]
