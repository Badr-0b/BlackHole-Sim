"""
blackhole_lens_interactive2.py
Interactive GPU gravitational-lens demo with:
 - finite gridded plane (lines) below the sphere
 - accretion disk (orange -> yellow) around the sphere
 - spherical lensing region around the black hole
 - mouse-left-drag orbit camera
 - mouse-wheel zoom
 - keys [ and ] to change LENS_MASS
Requirements:
    pip install PyOpenGL PyOpenGL_accelerate glfw numpy
"""
import glfw
from OpenGL.GL import *
import numpy as np
import time
import sys

# ----- settings -----
WINDOW_W, WINDOW_H = 960, 540
RENDER_W, RENDER_H = 360, 202   # internal low-res; tweak for perf
SPHERE_POS = np.array([0.0, 0.6, 1.0], dtype=np.float32)
SPHERE_RADIUS = 0.28
LENS_MASS = 1.5
LENS_RADIUS = 1.6     # spherical radius around BH where lensing is active
PLANE_SIZE = 6.0      # finite plane half-extent (plane is square: [-PLANE_SIZE,PLANE_SIZE])
GRID_SPACING = 0.25   # spacing between grid lines
DISK_INNER = 0.35
DISK_OUTER = 0.9
# ---------------------

# camera orbit control
yaw, pitch = 0.0, 12.0
distance = 2.2
last_x, last_y = 0.0, 0.0
mouse_down = False

VERTEX_SHADER_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main(){
    vUV = (aPos + 1.0)*0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform vec2 iResolution;
uniform float iTime;
uniform vec3 spherePos;
uniform float sphereRadius;
uniform float lensMass;
uniform float lensRadius;
uniform float diskInner;
uniform float diskOuter;
uniform float planeSize;
uniform float gridSpacing;
uniform vec3 camPos;
uniform vec3 camTarget;

#define PI 3.141592653589793

// cheap hash
float hash21(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+78.233); return fract(p.x*p.y); }

// ray direction from camera through pixel
vec3 rayDirFromPixel(vec2 uv, vec3 camPos, vec3 camTarget){
    vec2 screen = (uv - 0.5) * vec2(iResolution.x / iResolution.y, 1.0) * 1.1;
    vec3 forward = normalize(camTarget - camPos);
    vec3 right = normalize(cross(forward, vec3(0,1,0)));
    vec3 up = normalize(cross(right, forward));
    float fov = 1.0;
    return normalize(forward + screen.x*right*fov + screen.y*up*fov);
}

// intersect ray with plane y = planeY
bool intersectPlaneY(vec3 ro, vec3 rd, float planeY, out vec3 hitPos, out float t){
    float denom = rd.y;
    if(abs(denom) < 1e-5) { t = -1.0; hitPos = vec3(0); return false; }
    t = (planeY - ro.y) / denom;
    if(t <= 0.0){ hitPos = vec3(0); return false; }
    hitPos = ro + rd * t;
    return true;
}

// ray-sphere intersection
bool intersectSphere(vec3 ro, vec3 rd, vec3 center, float r, out float t0){
    vec3 L = ro - center;
    float a = dot(rd, rd);
    float b = 2.0 * dot(rd, L);
    float c = dot(L,L) - r*r;
    float disc = b*b - 4.0*a*c;
    if(disc < 0.0){ t0 = -1.0; return false; }
    float sq = sqrt(disc);
    float tA = (-b - sq) / (2.0*a);
    float tB = (-b + sq) / (2.0*a);
    t0 = (tA > 0.0) ? tA : ((tB > 0.0) ? tB : -1.0);
    return t0 > 0.0;
}

// point-mass-like deflection applied within lensRadius (visual approximation).
// rd_in is incident ray direction; ro is ray origin; center is mass center.
vec3 applyLensing(vec3 ro, vec3 rd_in, vec3 center, float mass, float lensR){
    // find closest approach parameter t_closest
    float t_closest = dot(center - ro, rd_in); // projection along rd_in
    vec3 closest = ro + rd_in * t_closest;
    vec3 bvec = closest - center; // vector from mass center to closest approach (perp to rd_in)
    float b = length(bvec) + 1e-6;
    // if closest approach is outside lensR (sphere), fade effect
    float dcenter = length(closest - center);
    float regionFactor = smoothstep(lensR, lensR*0.6, dcenter); // 1 near center -> 0 outside
    // direction to deflect: toward mass center projected perpendicular to ray
    vec3 defDir = normalize(bvec); // points away from center, we want toward center => negative
    // magnitude ~ mass / b (visual), scaled and attenuated
    float strength = mass / (b); // dimensionless visual
    // amplify near center
    strength *= (1.0 + 4.0 * (1.0/(1.0 + pow(b / (0.5*max(lensR,0.1)), 1.5))));
    // turn strength into small angle change along direction perpendicular to rd_in
    // create small delta to rd_in
    vec3 delta = -defDir * (strength * 0.02) * regionFactor;
    vec3 rd_out = normalize(rd_in + delta);
    return rd_out;
}

vec3 tonemap(vec3 c){
    c = c / (c + vec3(1.0));
    c = pow(c, vec3(1.0/2.2));
    return c;
}

void main(){
    vec2 uv = vUV;
    vec2 fragCoord = uv * iResolution;

    // camera ray (incident)
    vec3 rd = rayDirFromPixel(uv, camPos, camTarget);
    vec3 ro = camPos;

    // compute lensed ray (apply lensing if ray passes near sphere)
    vec3 rd_lensed = applyLensing(ro, rd, spherePos, lensMass, lensRadius);

    // First: check if ray (original camera ray) hits the sphere silhouette
    float t_sphere;
    bool hitSphere = intersectSphere(ro, rd, spherePos, sphereRadius, t_sphere);

    // Now compute plane intersection using the LENSED ray (so background & disk are bent)
    vec3 planeHit;
    float t_plane;
    bool planeOK = intersectPlaneY(ro, rd_lensed, 0.0, planeHit, t_plane);

    // plane color default background black
    vec3 planeColor = vec3(0.0);

    if(planeOK){
        // check if hit inside finite plane bounds
        if(abs(planeHit.x - spherePos.x) <= planeSize && abs(planeHit.z - spherePos.z) <= planeSize){
            // grid lines: line when coordinate near multiples of gridSpacing
            vec2 local = (planeHit.xz - spherePos.xz);
            // transform coordinates so grid lines are spaced by gridSpacing
            vec2 g = local / gridSpacing;
            vec2 gv = abs(fract(g) - 0.5);
            // thickness control via fwidth
            float fx = fwidth(g.x);
            float fy = fwidth(g.y);
            float lineX = smoothstep(0.0, 0.4*fx, gv.x);
            float lineY = smoothstep(0.0, 0.4*fy, gv.y);
            float lineMask = 1.0 - min(lineX, lineY); // 1 on lines
            // make lines bright white, otherwise near-black plane
            vec3 base = vec3(0.02); // plane base darkness
            vec3 lineCol = vec3(1.0);
            planeColor = mix(base, lineCol, lineMask);
            // slight falloff near sphere (shadow from the sphere)
            float d2 = length(planeHit.xz - spherePos.xz);
            float shadow = smoothstep(sphereRadius*1.1, sphereRadius*2.2, d2);
            planeColor *= mix(0.25, 1.0, shadow);
        } else {
            planeColor = vec3(0.0); // outside plane -> background black
        }
    }

    // Accretion disk: intersect disk plane at y = spherePos.y using LENSED ray
    vec3 diskColor = vec3(0.0);
    vec3 diskHit;
    float t_disk;
    if(intersectPlaneY(ro, rd_lensed, spherePos.y, diskHit, t_disk)){
    // compute cylindrical radius around sphere center
    vec2 rel = diskHit.xz - spherePos.xz;
    float r = length(rel);
    if(r >= diskInner && r <= diskOuter){
        // smooth edge
        float band = smoothstep(diskInner, diskInner + 0.02, r) - smoothstep(diskOuter - 0.02, diskOuter, r);
        
        // --- UPDATED ORANGE-HEAVY GRADIENT ---
        vec3 c0 = vec3(1.0, 0.45, 0.05);  // deep orange
        vec3 c1 = vec3(1.0, 0.7, 0.1);    // lighter orange (yellow accent)
        vec3 col = mix(c0, c1, smoothstep(diskInner, diskOuter, r));
        // --------------------------------------

        // fast spinning / brightness modulation
        float theta = atan(rel.y, rel.x);
        float spinSpeed = 5000.0;  // radians/sec (super fast)
        float dop = 0.7 + 0.7 * cos(theta + iTime * spinSpeed);
        diskColor = col * dop * band;
    }
}


    // Composite: background is black; planeColor + diskColor over it.
    vec3 col = vec3(0.0);
    col = planeColor + diskColor;

    // If camera ray hits sphere, draw sphere silhouette ON TOP (nearly black with rim)
    if(hitSphere){
        // sphere shading rim
        vec3 p = ro + rd * t_sphere;
        vec3 n = normalize(p - spherePos);
        float rim = pow(max(0.0, dot(n, normalize(vec3(0.0,1.0,1.0)))), 10.0);
        vec3 sphereCol = vec3(0.01) + 0.4 * rim * vec3(0.35, 0.18, 0.08);
        col = sphereCol;
    }

    // subtle vignette + noise
    float dcenter = length((fragCoord - iResolution * 0.5) / iResolution.y);
    col *= 1.0 - 0.45 * smoothstep(0.6, 1.0, dcenter);
    col += 0.003 * hash21(fragCoord.xy);

    col = tonemap(col);
    fragColor = vec4(col, 1.0);
}
"""

# --- GL helper functions ---
def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def link_program(vs, fs):
    vs_id = compile_shader(vs, GL_VERTEX_SHADER)
    fs_id = compile_shader(fs, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs_id)
    glAttachShader(prog, fs_id)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    glDeleteShader(vs_id)
    glDeleteShader(fs_id)
    return prog

# --- input callbacks ---
def mouse_button_callback(window, button, action, mods):
    global mouse_down
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_down = (action == glfw.PRESS)

def cursor_pos_callback(window, xpos, ypos):
    global last_x, last_y, yaw, pitch
    if mouse_down:
        dx = xpos - last_x
        dy = ypos - last_y
        yaw += dx * 0.25
        pitch -= dy * 0.25
        pitch = np.clip(pitch, -85.0, 85.0)
    last_x, last_y = xpos, ypos

def scroll_callback(window, xoffset, yoffset):
    global distance
    # zoom: scroll up (positive y) -> decrease distance (zoom in)
    distance *= (0.95 ** yoffset)
    distance = np.clip(distance, 0.6, 6.0)
    # print distance for feedback
    # print(f"distance = {distance:.3f}")

def key_callback(window, key, scancode, action, mods):
    global LENS_MASS
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_LEFT_BRACKET:
            LENS_MASS = max(0.05, LENS_MASS - 0.1)
        elif key == glfw.KEY_RIGHT_BRACKET:
            LENS_MASS = min(12.0, LENS_MASS + 0.1)
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        print(f"LENS_MASS = {LENS_MASS:.2f}, distance = {distance:.2f}")

# --- main ---
def main():
    global yaw, pitch, distance
    if not glfw.init():
        print("GLFW init failed")
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(WINDOW_W, WINDOW_H, "Black Hole Lensing (Interactive v2)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return
    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.swap_interval(1)

    prog = link_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)

    # fullscreen quad
    quad = np.array([-1.0,-1.0, 1.0,-1.0, -1.0,1.0, 1.0,1.0], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    glUseProgram(prog)
    locs = {name: glGetUniformLocation(prog, name) for name in [
        "iResolution","iTime","spherePos","sphereRadius","lensMass","lensRadius",
        "diskInner","diskOuter","planeSize","gridSpacing","camPos","camTarget"
    ]}

    start = time.time()
    while not glfw.window_should_close(window):
        t = time.time() - start

        # compute camera pos from yaw/pitch/distance (spherical)
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        cx = distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        cy = distance * np.sin(pitch_rad) + 0.6
        cz = distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        cam_pos = np.array([cx, cy, -cz], dtype=np.float32)
        cam_target = np.array([SPHERE_POS[0], SPHERE_POS[1], SPHERE_POS[2]], dtype=np.float32)

        w, h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, w, h)
        glClearColor(0.0,0.0,0.0,1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(prog)
        glUniform2f(locs["iResolution"], float(RENDER_W), float(RENDER_H))
        glUniform1f(locs["iTime"], float(t))
        glUniform3f(locs["spherePos"], float(SPHERE_POS[0]), float(SPHERE_POS[1]), float(SPHERE_POS[2]))
        glUniform1f(locs["sphereRadius"], float(SPHERE_RADIUS))
        glUniform1f(locs["lensMass"], float(LENS_MASS))
        glUniform1f(locs["lensRadius"], float(LENS_RADIUS))
        glUniform1f(locs["diskInner"], float(DISK_INNER))
        glUniform1f(locs["diskOuter"], float(DISK_OUTER))
        glUniform1f(locs["planeSize"], float(PLANE_SIZE))
        glUniform1f(locs["gridSpacing"], float(GRID_SPACING))
        glUniform3f(locs["camPos"], float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        glUniform3f(locs["camTarget"], float(cam_target[0]), float(cam_target[1]), float(cam_target[2]))

        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        glfw.terminate()
        sys.exit(1)
