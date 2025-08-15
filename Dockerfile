FROM python:3.11-slim

WORKDIR /app

# copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all from cwd to the workdir set in container before
COPY . .

# huggingface spaces needs port 7860
EXPOSE 7860

# run app.py with python interpreter if container is run
CMD ["python", "app.py"]


