FROM python:3.11-slim-bullseye

WORKDIR /src
RUN pip install --no-cache-dir poetry
COPY pyproject.toml /src/pyproject.toml
RUN poetry install
COPY . /src

ENTRYPOINT ["poetry", "run"]