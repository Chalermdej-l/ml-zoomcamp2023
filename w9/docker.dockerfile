FROM agrigorev/zoomcamp-bees-wasps:v2

COPY predict.py .

RUN pip install tensorflow
RUN pip install numpy
RUN pip install Pillow
RUN pip install scipy

CMD [ "python","predict.py" ]