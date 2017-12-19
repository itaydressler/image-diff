FROM jjanzic/docker-python3-opencv:contrib-opencv-3.3.0

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 80

CMD [ "python", "./api.py" ]