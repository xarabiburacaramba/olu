FROM conda/miniconda3

RUN conda update conda \
 && conda install -c conda-forge  -y \
    flask\
    flask-restplus \
    gdal=2.4.1 \
    libgdal \
    psycopg2 \
    pillow \
    rtree \
    werkzeug \
 && conda clean -tipsy

RUN apt-get update && apt-get install -y build-essential  
RUN pip install cython
RUN pip install  \
    flask-session \
    pygeoprocessing

WORKDIR /app
COPY src /app

ENV FLASK_ENV development

EXPOSE 5000

RUN adduser --disabled-password myuser
USER myuser

ENTRYPOINT ["python"]

# Run app.py when the container launches
CMD ["app.py"]


