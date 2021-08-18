FROM continuumio/miniconda3:latest

WORKDIR /code
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
ENV PYTHONPATH=$PYTHONPATH:/app:/app/src


WORKDIR /work

# ENTRYPOINT ["ls", "-latr"]

ENTRYPOINT [ "/opt/conda/bin/python3", "/app/update.py" ]
CMD ["--all", "--config", "config.json"]

