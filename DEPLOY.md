# Deployment Guide: Blackhole Simulation on Remote Linux VM

This guide explains how to run the `blackhole.py` OpenGL simulation inside a Docker container on a remote Linux machine (with NVIDIA GPU) and view the output on your local machine using X11 forwarding.

## Prerequisites

1.  **Remote Machine (Linux)**:
    *   NVIDIA GPU installed.
    *   **NVIDIA Drivers** installed (verify with `nvidia-smi`).
    *   **Docker** installed.
    *   **NVIDIA Container Toolkit** installed (allows Docker to access the GPU).
        *   Test readiness: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

2.  **Local Machine (Client)**:
    *   An SSH client that supports X11 forwarding.
    *   **X Server** running locally:
        *   **Windows**: [VcXsrv](https://sourceforge.net/projects/vcxsrv/) (recommended) or Xming.
        *   **macOS**: [XQuartz](https://www.xquartz.org/).
        *   **Linux**: Native X11 usually works out of the box.

---

## Step 1: Connect with X11 Forwarding

When connecting to your remote machine, you must enable X11 forwarding.

**Command Line (SSH):**
```bash
# The -X flag enables X11 forwarding.
# If -X doesn't work, try -Y (trusted X11 forwarding).
ssh -X user@remote-ip-address
```

> **For Windows Users (PuTTY/Windows Terminal):**
> *   If using **Windows Terminal** or PowerShell with `ssh`, ensure VcXsrv is running first.
> *   If using **PuTTY**: Go to `Connection > SSH > X11` and check "Enable X11 forwarding".

## Step 2: Build the Docker Image

On the **remote machine**, navigate to the project directory and build the image:

```bash
cd /path/to/blackhole
docker build -t blackhole .
```

## Step 3: Run the Application

To run the container, we need to:
1.  Pass the `--gpus all` flag.
2.  Pass the `DISPLAY` environment variable.
3.  Mount the X11 socket (to allow communication with your local display).
4.  Share the `.Xauthority` file (authentication for the display).

Run the following command:

```bash
docker run -it --rm \
    --gpus all \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    blackhole
```

### Explanation of Flags:
*   `--gpus all`: Grants the container access to the NVIDIA GPU.
*   `--net=host`: Shares the host's network namespace (often helps with X11 connectivity on some setups, though not strictly always required it simplifies things).
*   `-e DISPLAY=$DISPLAY`: Passes your current X11 display ID (e.g., `localhost:10.0`) to the container.
*   `-v /tmp/.X11-unix:...`: Maps the X11 socket.
*   `-v $HOME/.Xauthority:...`: Maps the authentication token so the root user inside the container can talk to your X server.

## Troubleshooting

*   **"Error: Can't open display"**:
    *   Run `echo $DISPLAY` on the remote host outside docker. It should look like `localhost:10.0`.
    *   Ensure your local X Server (VcXsrv/XQuartz) is running.
    *   If using VcXsrv on Windows, ensure "Disable access control" is checked in the launch settings if you have trouble connecting.

*   **"Glfw Error 65544: X11: The DISPLAY environment variable is missing"**:
    *   Double-check that `-e DISPLAY=$DISPLAY` is passing a value.
    *   Try running `xhost +` on the remote host (temporarily allows all connections) to test if it's a permission issue. (Use carefully).
