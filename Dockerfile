FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Expose HTTPS port
EXPOSE 443
COPY ./app /app/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "443", "--ssl-keyfile", "/app/app/private_key.key", "--ssl-certfile", "/app/app/certificate.crt"]