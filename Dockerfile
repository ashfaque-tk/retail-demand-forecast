FROM python:3.11-slim

#Bring the uv binary from Astral's official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System-level dependency LightGBM needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app 

#dependencies first 
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# Now the actual application
COPY main.py .
COPY src/ ./src/
COPY models/ ./models/
COPY data/raw/calendar.csv ./data/raw/calendar.csv
COPY data/raw/sell_prices.csv ./data/raw/sell_prices.csv

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
