FROM python:3.13-slim

# Install git for cloning the repository
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files to the container
COPY . /app

# Install with pip using the repository URL
RUN python -m pip install .

# Set the default command
CMD ["/bin/bash", "--login"]
