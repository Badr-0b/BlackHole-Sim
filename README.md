# Black Hole Gravitational Lensing — Interactive GPU Simulation
### Real-time OpenGL shader rendering inspired by *Interstellar’s* black hole scene

This project renders a physically-inspired gravitational lensing simulation in real time using a custom GLSL fragment shader. The effect models how light bends around a massive rotating black hole, producing warped spacetime visuals and an orbiting accretion disk — fully interactive with camera orbit, zoom, and lens mass control.

---
## Update: Docker support for linux-based users has been added. please scroll down to the bottom of this README for instructions.
---

##  Overview
This application demonstrates how GPU ray-marching, shader-based lens distortion, and OpenGL rendering can be combined to visualize extreme gravitational effects. The black hole's mass controls the intensity of light deflection. Users can freely rotate the camera, zoom, and adjust simulation parameters to observe the lensing effect from different viewpoints.

Originally created as a graphics & physics learning project, this evolved into a full interactive simulation suitable for portfolios and research demonstrations.

---

##  Key Features
- Real-time GLSL gravitational lensing distortion
- Rotating accretion disk simulation with Doppler-shift color modulation
- Ray-based plane intersection grid backdrop
- Fully interactive camera (orbit + zoom)
- Adjustable black hole mass in real time
- Runs natively on OpenGL 3.3 hardware

---

##  Tech Stack
| Layer | Technology |
|-------|------------|
| Rendering | OpenGL 3.3, GLSL |
| Windowing / Input | GLFW |
| Python Bindings | PyOpenGL, PyOpenGL_accelerate |
| Math & Utilities | NumPy |
| Platform | Desktop (Windows / Linux / macOS if OpenGL 3.3 supported) |

---

##  Installation
### Requirements
```bash
Python 3.8+
GPU with OpenGL 3.3+
```
---
## Dependencies
```bash
pip install PyOpenGL PyOpenGL_accelerate glfw numpy
```

---

## Usage Guide

| Action             | Control      |
| ------------------ | ------------ |
| Orbit camera       | Click + drag |
| Zoom               | Scroll wheel |
| Increase lens mass | `]`          |
| Decrease lens mass | `[`          |
| Exit               | `Esc`        |

Variables you can tweak in `FRAGMENT_SHADER_SRC`:
| Name           | Effect                               |
| -------------- | ------------------------------------ |
| `spinSpeed`    | Rotation speed of the accretion disk |
| `lensMass`     | Gravitational lens strength          |
| `lensRadius`   | Size of the distortion region        |
| `sphereRadius` | Radius of the singularity sphere     |

---

### **Running the Black Hole Simulation (Linux)**

1. Build the Docker image:

```bash
docker build -t blackhole-opengl .
```

2. Allow Docker access to X11:

```bash
xhost +local:docker
```

3. Run the simulation:

```bash
docker run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/dri:/dev/dri \
    blackhole-opengl
```

4. After running, optionally remove X11 access:

```bash
xhost -local:docker
```

---

Last Updated: 4 Dec 2025
