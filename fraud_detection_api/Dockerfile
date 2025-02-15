# Stage 1: Build dependencies (build environment)
FROM python:3.10.16-slim as build-stage

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH so that the scripts folder is found
ENV PYTHONPATH=/app

# Expose port 5000
EXPOSE 5000

# Run the API
CMD ["python", "serve_model_api.py"]
