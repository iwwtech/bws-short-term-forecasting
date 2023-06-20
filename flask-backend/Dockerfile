# Stage 1 - install dependencies
FROM python:3.7-bullseye as builder

COPY requirements.txt /requirements.txt

RUN pip install --user -r /requirements.txt

# Stage 2 - Start from clean image without cached files etc...
# Keep only the libraries installed in the previous stage
# Note: Using a smaller base image like Alpine leads to uninformativ errors
FROM python:3.7-bullseye

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

EXPOSE 5000

WORKDIR /core-tool

COPY . /core-tool

CMD ["python", "run.py"]
