# Use a lightweight Python image
FROM python:3.10.16-slim as build-stage

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the API
CMD ["python", "serve_model_api.py"]
