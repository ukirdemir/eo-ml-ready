FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
# Add ",eumetsat" to read SEVIRI (satpy and eumdac make the image much larger).
RUN pip install --no-cache-dir -e ".[train,track,advection]"

ENV PYTHONUNBUFFERED=1
VOLUME ["/app/data", "/app/artifacts"]

ENTRYPOINT ["python", "-m"]
CMD ["eo_ml_ready.train", "--help"]
