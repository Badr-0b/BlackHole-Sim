# Architecture Overview

This project follows a minimal GPU-driven rendering pipeline. Nearly all visual
logic is executed in the **fragment shader**, not Python.

## Rendering Strategy
| Stage | Responsibility |
|-------|----------------|
| Python / CPU | Setup, input, camera transforms |
| GPU Vertex Shader | Draws a fullscreen quad |
| GPU Fragment Shader | Computes gravitational lensing per-pixel |

The simulation does **not** rasterize meshes into 3D space. Instead, each pixel
computes a camera ray; that ray is bent around the black hole analytically, then
used to determine hit color (grid plane, accretion disk, or singularity).

## Why a fullscreen quad?
The computation is per-pixel, not per-vertex. Rendering a quad full-screen allows
the shader to treat every pixel like a ray-tracing sample.

## Camera Model
The camera is orbit-style:
- Yaw and pitch from mouse drag
- Radius from scroll wheel
- Cemtered around the "black hole"'s position

## Lensing Model
Not a general relativity solver — instead a tuned visual approximation:
- Computes distance of ray to singularity
- Applies directional perturbation proportional to mass and impact parameter
- Normalizes result to maintain valid ray direction


LLMs helped me learn SO much. i absolutely DESPISE the controls section, shouldve left it at just raw visual simulation.
