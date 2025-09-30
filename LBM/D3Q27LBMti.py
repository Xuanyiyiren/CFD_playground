import taichi as ti
import numpy as np
from matplotlib import cm
from tqdm import tqdm
import os
import shutil
real = ti.f32
cmap_name = "magma_r"  # python colormap

Nx = 400
Ny = 100
Nz = 100
# Nx = 16
# Ny = 16
tau = .53

Nv = 27
# Use dtype instead of deprecated dt, and keep integer types for lattice velocities
 # D3Q27 速度方向编号说明（与下方 directions/veloI/veloJ/veloK 一一对应）：
 # 0:  (0, 0, 0)  - 中心节点
 # 1:  (+1, 0, 0) - 面心 +x
 # 2:  (-1, 0, 0) - 面心 -x
 # 3:  (0, +1, 0) - 面心 +y
 # 4:  (0, -1, 0) - 面心 -y
 # 5:  (0, 0, +1) - 面心 +z
 # 6:  (0, 0, -1) - 面心 -z
 # 边心（12）：
 # 7:  (+1, +1, 0)   8:  (-1, +1, 0)   9:  (+1, -1, 0)  10: (-1, -1, 0)
 # 11: (+1, 0, +1)  12: (-1, 0, +1)  13: (+1, 0, -1)  14: (-1, 0, -1)
 # 15: (0, +1, +1)  16: (0, -1, +1)  17: (0, +1, -1)  18: (0, -1, -1)
 # 角点（8）：
 # 19: (+1, +1, +1) 20: (-1, +1, +1) 21: (+1, -1, +1) 22: (-1, -1, +1)
 # 23: (+1, +1, -1) 24: (-1, +1, -1) 25: (+1, -1, -1) 26: (-1, -1, -1)
 # 使用紧凑定义，避免逐项刷屏，同时保留完整编号注释
 # D3Q27 方向：0 中心，1–6 面心，7–18 边心，19–26 角点
directions = [
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
    (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
]
veloI = ti.Vector([d[0] for d in directions], dt = ti.i32)
veloJ = ti.Vector([d[1] for d in directions], dt = ti.i32)
veloK = ti.Vector([d[2] for d in directions], dt = ti.i32)

# D3Q27 权重（按 216 统一缩放为整数）：中心64，面心16，边心4，角点1
weights = [64.0] + [16.0]*6 + [4.0]*12 + [1.0]*8
weights = ti.Vector(weights, dt = real)
weights /= sum(weights)

rho = 10.34
ux = 1.28 / rho
uy = 0 / rho
uz = 0 / rho

@ti.func
def setfeq(rho, ux, uy, uz):
    cdot = veloI * ux + veloJ * uy + veloK * uz
    return rho * weights * (
        1 + 3 * cdot + 9/2 * cdot**2 - 3/2 * (ux**2 + uy**2 + uz**2)
    )

# Geometry: circular cylinder obstacle (match numpy reference)
radius = 13
sphere = np.full((Nx, Ny, Nz), False)
center = (Nx // 4, Ny // 2, Nz // 2)
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2 < radius**2:
                sphere[i, j, k] = True

ti.init(arch=ti.cpu)
# cylinder_ti = ti.field(bool, shape=(Nx, Ny))
shpere_ti = ti.field(ti.u8, shape=(Nx, Ny, Nz))
shpere_ti.from_numpy(sphere.astype(np.uint8))
botzf = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny, Nz))
botzf_backup = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny, Nz))
primvars = ti.Vector.field(4, dtype=real, shape=(Nx, Ny, Nz)) # rho, ux, uy, uz
movingfeq = ti.Vector.field(Nv, dtype=real, shape=())
staticfeq = ti.Vector.field(Nv, dtype=real, shape=())
nan_detected = ti.field(dtype=ti.i32, shape=())  # flag for NaN detection

@ti.kernel
def check_nan() -> int:
    # Check if any NaN exists in botzf
    nan_count = 0
    for i, j, k in ti.ndrange(Nx, Ny, Nv):
        if ti.math.isnan(botzf[i, j][k]):
            nan_count += 1
    return nan_count

@ti.kernel
def initialize():
    movingfeq[None] = setfeq(rho, ux, uy, uz)
    staticfeq[None] = setfeq(rho, 0, 0, 0)
    for i, j, k in ti.ndrange(Nx, Ny, Nz):
        if shpere_ti[i, j, k]:
            botzf[i, j, k] = staticfeq[None]
        else:
            for v in ti.static(range(Nv)):
                # Add a small perturbation to trigger vortex shedding
                # Keep distributions positive by using tiny uniform noise
                botzf[i, j, k][v] = movingfeq[None][v] # + noise_amp * (ti.random(real))

@ti.kernel
def collision():
    for I in ti.grouped(ti.ndrange(Nx, Ny, Nz)):
        botzf_ = botzf[I]
        rho = botzf_.sum()
        ux = botzf_.dot(veloI) / rho
        uy = botzf_.dot(veloJ) / rho
        uz = botzf_.dot(veloK) / rho
        
        primvars[I] = [rho, ux, uy, uz] # store
        botzf_eq = setfeq(rho, ux, uy, uz)
        botzf_ = botzf_ + (botzf_eq - botzf_) / tau
        botzf[I] = botzf_

@ti.kernel
def copy_botzf():
    for I in ti.grouped(ti.ndrange(Nx, Ny, Nz)):
        botzf_backup[I] = botzf[I]

@ti.kernel
def streaming():
    for i, j, k in ti.ndrange(Nx, Ny, Nz):
        for v in ti.static(range(Nv)):
            botzf[i, j, k][v] = botzf_backup[(i - veloI[v] + Nx) % Nx,
                                             (j - veloJ[v] + Ny) % Ny,
                                             (k - veloK[v] + Nz) % Nz][v]

@ti.kernel
def boundary_condition():
    for i, j, k in ti.ndrange(Nx, Ny, Nz):
        if shpere_ti[i, j, k]:
            # bounce-back boundary condition
            botzf_ = botzf[i, j, k]
            botzf_[1], botzf_[2] = botzf_[2], botzf_[1]
            botzf_[3], botzf_[4] = botzf_[4], botzf_[3]
            botzf_[5], botzf_[6] = botzf_[6], botzf_[5]
            
            botzf_[7], botzf_[10] = botzf_[10], botzf_[7]
            botzf_[8], botzf_[9] = botzf_[9], botzf_[8]
            botzf_[11], botzf_[14] = botzf_[14], botzf_[11]
            botzf_[12], botzf_[13] = botzf_[13], botzf_[12]

            botzf_[19], botzf_[26] = botzf_[26], botzf_[19]
            botzf_[20], botzf_[25] = botzf_[25], botzf_[20]
            botzf_[21], botzf_[24] = botzf_[24], botzf_[21]
            botzf_[22], botzf_[23] = botzf_[23], botzf_[22]
            botzf[i, j, k] = botzf_
        # Inlet (left boundary): velocity inlet
        if i == 0:
            botzf[0, j, k] = movingfeq[None]
        # Outlet (right boundary): absobtion boundary condition
        if i == Nx - 1:
            for v in ti.static(
                # [1, 7, 9, 11, 13, 19, 21, 23, 25]
                [2, 8, 10, 12, 14, 20, 22, 24, 26]
            ):
                botzf[Nx - 1, j, k][v] = botzf[Nx - 2, j, k][v]
            # todo : filter out those outflow directions

#     # Pass 2: find global min/max
#     max_ = -1.0e30
#     min_ = 1.0e30
#     for i, j in ti.ndrange(Nx, Ny):
#         ti.atomic_max(max_, img[i, j])
#         ti.atomic_min(min_, img[i, j])

#     # Pass 3: normalize to [0,1]
#     for i, j in ti.ndrange(Nx, Ny):
#         denom = max_ - min_
#         val = 0.0
#         if denom > 1e-12:
#             val = (img[i, j] - min_) / denom
#         img[i, j] = val
                
# def main():
#     # Minimal GUI loop to render density (1) and velocity magnitude (2)
#     initialize()
#     gui = ti.GUI("LBM D2Q9", res=(Nx, Ny))
#     cmap = cm.get_cmap(cmap_name)
#     # Prepare ./simufigs: clear then recreate before simulation
#     out_dir = './simufigs'
#     shutil.rmtree(out_dir, ignore_errors=True)
#     os.makedirs(out_dir, exist_ok=True)
#     img_mode = 1  # 0: density, 1: |u|
#     it = 0
#     skip = 50  # render every 'skip' steps
#     max_steps = 15000  # maximum simulation steps
#     frame = 0
#     pbar = tqdm(total=max_steps, desc="Simulating", unit="step")
#     while gui.running and it < max_steps:
#         # Simulation step
#         copy_botzf()
#         streaming()
#         collision()
#         boundary_condition()
#         # Check for NaN values - stop if any detected
#         if check_nan() > 0:
#             print(f"NaN detected in botzf at step {it}. Simulation stopped.")
#             break

#         # Render
#         if it % skip == 0:
#             paint(img_mode)
#             gui.set_image(cmap(img.to_numpy()))
#             gui.show(os.path.join(out_dir, f'frame_{frame:05d}.png'))
#             frame += 1
#         it += 1
#         pbar.update(1)

#         # Simple key toggles: '1' for density, '2' for velocity magnitude
#         if gui.get_event():
#             if gui.event.key == '1':
#                 img_mode = 0
#             elif gui.event.key == '2':
#                 img_mode = 1
#     pbar.close()

# # def main():
# #     initialize()
# #     collision()
# #     streaming()
# #     boundary_condition()

# if __name__ == '__main__':
#     main()