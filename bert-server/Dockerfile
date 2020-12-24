FROM tensorflow/tensorflow:1.12.0-py3
RUN pip install --upgrade pip
RUN pip install bert-serving-server[http]
RUN mkdir -p /app
COPY ./entrypoint.sh /app
RUN chmod +x /app/entrypoint.sh
WORKDIR /app
ENTRYPOINT ["./entrypoint.sh"]
CMD []