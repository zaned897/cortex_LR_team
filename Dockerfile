
FROM ubuntu:20.04

#UPDATE AND UPGRADE
RUN apt update && apt upgrade -y

# INSTALL OCR (TESSERACT) PYTHON AND UTILS
RUN apt install poppler-utils tesseract-ocr -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt install python3 python3-pip -y

# COPPY THE APP

COPY /api /home/app/api

# copy the main scrips and configuration files
COPY /NPDB/main_npdb_cover_beta.py /home/app/NPDB/
COPY /NPDB/lossrun.py /home/app/NPDB/
COPY /NPDB/config /home/app/NPDB/config/

COPY /LOSSRUNS/main_lossruns.py /home/app/LOSSRUNS/
COPY /LOSSRUNS/lossrun.py /home/app/LOSSRUNS/
COPY /LOSSRUNS/config /home/app/LOSSRUNS/config/
COPY /data/ /home/app/data/

# copy the natural language models and the requirements
COPY /models /home/app/models
COPY /src/requirements.txt /home/app/src/

# copy unit test
COPY /test/ /home/app/test/
COPY /testLR/ /home/app/testLR/

# results folder for all tested files
COPY /results/ /home/app/results/

# INSTALL PYTHON PACKAGES
RUN pip3 install -r /home/app/src/requirements.txt

WORKDIR /home/app/
RUN pip3 install -r api/requirements.txt
CMD ["python3", "api/api.py"]
EXPOSE 8080

# WARNING: IT'S NECESASY TO COPY THE MODEL TO /HOME/APP/MODELS/
# THE USE FOR NPDB DEMO: python3 /home/app/NPDB/main_npdb_cover_beta.py [params]
