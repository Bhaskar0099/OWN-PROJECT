
# Use Python 3.11.7 as the base image
FROM python:3.11.7-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY app.py .

# Copy .env file with API key
COPY .env .

# Copy data directory
COPY data/ ./data/

# Expose the port the app runs on
EXPOSE 8084

# Command to run the application
CMD ["python", "app.py"]