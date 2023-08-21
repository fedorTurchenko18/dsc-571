FROM arm64v8/python:3.10-slim

RUN mkdir dsc-571
WORKDIR /dsc-571

RUN mkdir service
WORKDIR /dsc-571/service

RUN mkdir app_ui
WORKDIR /dsc-571/service/app_ui

COPY . .

RUN pip install -r requirements.txt

CMD [ "gunicorn", "--threads=1", "-b 0.0.0.0:80", "app:server"]