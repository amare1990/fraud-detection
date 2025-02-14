# Stage 1: Build dependencies (build environment)
FROM python:3.10.16-slim as build-stage

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Set the entrypoint to run your API app (assuming it is serve_model_api.py)
CMD ["python", "serve_model_api.py"]
