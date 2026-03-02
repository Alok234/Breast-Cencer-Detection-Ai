# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start backend with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
