FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY server.py policy.py ./
EXPOSE 8080
CMD ["python", "policy.py"]
