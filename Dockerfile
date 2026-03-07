# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 4567

# Set environment variables for Flask
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["python", "drive.py"]