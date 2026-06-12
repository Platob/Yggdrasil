FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[api,http]"

COPY . .

CMD ["uvicorn", "services.bot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
