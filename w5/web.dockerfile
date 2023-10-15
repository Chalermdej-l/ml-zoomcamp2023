FROM svizor/zoomcamp-model:3.10.12-slim
# add your stuff here

COPY requirement.txt .

RUN pip install -r  requirement.txt

COPY ./web.py ./web.py
COPY dv.bin .
COPY model1.bin .