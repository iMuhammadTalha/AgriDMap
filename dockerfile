FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run migrations
RUN python manage.py migrate

# Expose the app port
EXPOSE 8008

# Start the app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8008"]
