FROM python:3.10.14


# Copy your Python application code into the container
COPY . ./app

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run your Python application
CMD ["python3", "main.py"]
