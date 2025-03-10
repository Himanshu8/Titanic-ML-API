# Use an official Python runtime as a parent image
FROM python:3.13

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY train_model.py train_model.py
COPY model.pkl model.pkl
COPY templates/ templates/
COPY static/ static/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]
