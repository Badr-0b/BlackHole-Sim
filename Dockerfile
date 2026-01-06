# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
# python3-pip: for installing python packages
# libglfw3: generic windowing support
# libgl1-mesa-dev, libglu1-mesa-dev: OpenGL libraries
# xorg-dev, libx11-dev: X11 support for window forwarding
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglfw3 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libx11-dev \
    xorg-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY blackhole.py .

# Command to run the application
# python3 -u (unbuffered output)
CMD ["python3", "-u", "blackhole.py"]
