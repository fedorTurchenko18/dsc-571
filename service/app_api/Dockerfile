FROM arm64v8/python:3.10-slim

RUN apt-get update \
    && apt-get install -y build-essential

WORKDIR /dsc-571

RUN mkdir service
RUN mkdir service/app_api

COPY . /dsc-571/service/app_api

RUN pip install -r /dsc-571/service/app_api/requirements.txt

CMD ["uvicorn", "service.app_api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]