import taichi as ti
import numpy as np
from matplotlib import cm
from tqdm import tqdm
import os
import shutil
import pyvista as pv
real = ti.f32
cmap_name = "magma_r"  # python colormap

Nx = 400
Ny = 200
# Nx = 16
# Ny = 16
tau = .520 # .510

Nv = 9
# Use dtype instead of deprecated dt, and keep integer types for lattice velocities
veloI = ti.Vector([0, 0, 1, 0, -1, 1, 1, -1, -1], dt=ti.i32)
veloJ = ti.Vector([0, 1, 0, -1, 0, 1, -1, -1, 1], dt=ti.i32)
weights = ti.Vector([16.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0], dt=real) / 36.0

rho_ref = 1
rho_ave = 1
# drho_ref = rho_ref - rho_ave
ux_ref = 0.1
uy_ref = 0

@ti.func
def setfeq(rho, ux, uy):
    cdot = veloI * ux + veloJ * uy
    return rho * weights * (
        1 + 3 * cdot + 9/2 * cdot**2 - 3/2 * (ux**2 + uy**2)
    ) - weights * rho_ave

# movingfeq = setfeq(rho, ux, uy)
# staticfeq = setfeq(rho, 0, 0)

# Geometry: circular cylinder obstacle (match numpy reference)
radius = 13
cylinder = np.full((Nx, Ny), False)
center = (Nx // 4, Ny // 2)
for i in range(Nx):
    for j in range(Ny):
        if (i - center[0])**2 + (j - center[1])**2 < radius**2:
            cylinder[i, j] = True

ti.init(arch=ti.gpu)
# cylinder_ti = ti.field(bool, shape=(Nx, Ny))
cylinder_ti = ti.field(ti.u8, shape=(Nx, Ny))
cylinder_ti.from_numpy(cylinder.astype(np.uint8))
botzf = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny), layout=ti.Layout.AOS)
botzf_prev = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny), layout=ti.Layout.AOS)
primvars = ti.Vector.field(3, dtype=real, shape=(Nx, Ny), layout=ti.Layout.AOS) # rho, ux, uy
# botzf = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny), layout=ti.Layout.SOA)
# botzf_prev = ti.Vector.field(Nv, dtype=real, shape=(Nx, Ny), layout=ti.Layout.SOA)
# primvars = ti.Vector.field(3, dtype=real, shape=(Nx, Ny), layout=ti.Layout.SOA) # rho, ux, uy
movingfeq = ti.Vector.field(Nv, dtype=real, shape=())
staticfeq = ti.Vector.field(Nv, dtype=real, shape=())
img = ti.field(dtype=real, shape=(Nx, Ny))  # image buffer for rendering
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
    movingfeq[None] = setfeq(rho_ref, ux_ref, uy_ref)
    staticfeq[None] = setfeq(rho_ref, 0, 0)
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder_ti[i, j]:
            botzf[i, j] = staticfeq[None]
        else:
            for k in ti.static(range(Nv)):
                # Add a small perturbation to trigger vortex shedding
                # Keep distributions positive by using tiny uniform noise
                botzf[i, j][k] = movingfeq[None][k] # + noise_amp * (ti.random(real))

# @ti.kernel
# def initialize():
#     # movingfeq[None] = setfeq(rho, ux, uy)
#     staticfeq[None] = setfeq(rho, 0, 0)
#     for i, j in ti.ndrange(Nx, Ny):
#         botzf[i, j] = staticfeq[None]

@ti.kernel
def collision():
    for i, j in ti.ndrange(Nx, Ny):
        botzf_ = botzf[i, j]
        drho = botzf_.sum() 
        rho = drho + rho_ave
        ux = botzf_.dot(veloI) / rho
        uy = botzf_.dot(veloJ) / rho
        primvars[i, j] = [rho, ux, uy] # store
        # botzf_eq = setfeq(drho, ux, uy)
        botzf_eq = setfeq(rho, ux, uy)
        botzf_ = botzf_ + (botzf_eq - botzf_) / tau
        botzf[i, j] = botzf_

@ti.kernel
def copy_botzf():
    for i, j in ti.ndrange(Nx, Ny):
        botzf_prev[i, j] = botzf[i, j]

@ti.kernel
def streaming():
    for i, j in ti.ndrange(Nx, Ny):
        for v in ti.static(range(Nv)):
            botzf[i, j][v] = botzf_prev[(i - veloI[v] + Nx) % Nx, (j - veloJ[v] + Ny) % Ny][v]

@ti.kernel
def boundary_condition():
    for i, j in ti.ndrange(Nx, Ny):
        if cylinder_ti[i, j]:
            # bounce-back boundary condition
            botzf_ = botzf[i, j]
            botzf_[1], botzf_[3] = botzf_[3], botzf_[1]
            botzf_[2], botzf_[4] = botzf_[4], botzf_[2]
            botzf_[5], botzf_[7] = botzf_[7], botzf_[5]
            botzf_[6], botzf_[8] = botzf_[8], botzf_[6]
            botzf[i, j] = botzf_
        # Inlet (left boundary): impose equilibrium with target (rho, ux, uy)
        if i == 0:
            botzf[0, j] = movingfeq[None]
            # botzf[0, j][2] = movingfeq[None][2]
            # botzf[0, j][5] = movingfeq[None][5]
            # botzf[0, j][6] = movingfeq[None][6]
        # Outlet (right boundary): simple zero-gradient copy from the previous column
        if i == Nx - 1:
            # botzf[Nx - 1, j][1] = botzf[Nx - 2, j][1]
            # botzf[Nx - 1, j][3] = botzf[Nx - 2, j][3]
            botzf[Nx - 1, j][4] = botzf[Nx - 2, j][4]
            botzf[Nx - 1, j][7] = botzf[Nx - 2, j][7]
            botzf[Nx - 1, j][8] = botzf[Nx - 2, j][8]

@ti.kernel
def paint(img_mode: int):
    # Pass 1: write raw scalar into img based on mode (0: density, 1: |u|)
    for i, j in ti.ndrange(Nx, Ny):
        rho_ = primvars[i, j][0]
        ux_ = primvars[i, j][1]
        uy_ = primvars[i, j][2]
        val = rho_
        if img_mode == 1:
            val = ti.sqrt(ux_ * ux_ + uy_ * uy_)
        img[i, j] = val

    # Pass 2: find global min/max
    max_ = -1.0e30
    min_ = 1.0e30
    for i, j in ti.ndrange(Nx, Ny):
        ti.atomic_max(max_, img[i, j])
        ti.atomic_min(min_, img[i, j])

    # Pass 3: normalize to [0,1]
    for i, j in ti.ndrange(Nx, Ny):
        denom = max_ - min_
        val = 0.0
        if denom > 1e-12:
            val = (img[i, j] - min_) / denom
        img[i, j] = val

def write_vtk(step: int, out_dir: str = './vtk'):
    """使用 pyvista 生成并保存 VTK XML（.vti，ImageData）文件。

    输出为 ImageData（VTK STRUCTURED_POINTS/UniformGrid 等价），对应点数据包含：
    - density: float32 标量
    - velocity: float32 三分量向量 (ux, uy, 0)
    - obstacle: uint8 标量（障碍物掩码）
    """
    os.makedirs(out_dir, exist_ok=True)

    # 提取当前原始变量（rho, ux, uy）与障碍物掩码
    prim = primvars.to_numpy()  # (Nx, Ny, 3)
    rho_np = prim[..., 0]
    ux_np = prim[..., 1]
    uy_np = prim[..., 2]
    obs_np = cylinder_ti.to_numpy()  # (Nx, Ny) -> 0/1

    # 构建 ImageData（与 STRUCTURED_POINTS/UniformGrid 等价，兼容旧版本 PyVista）
    grid = pv.ImageData()
    grid.dimensions = (Nx, Ny, 1)  # 点数（而非单元数）
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = (1.0, 1.0, 1.0)

    # 为与 VTK 点顺序一致：VTK 使用 x 变化最快、y 次之、z 最慢
    # 由于 numpy 数组形状为 (Nx, Ny)，直接使用 Fortran 顺序展平即可匹配 VTK 点顺序
    rho_flat = rho_np.flatten(order='F').astype(np.float32)
    ux_flat = ux_np.flatten(order='F').astype(np.float32)
    uy_flat = uy_np.flatten(order='F').astype(np.float32)
    obs_flat = obs_np.flatten(order='F').astype(np.uint8)

    # 写入点数据
    grid.point_data['density'] = rho_flat
    vel = np.column_stack(
        [ux_flat, uy_flat, np.zeros_like(ux_flat, dtype=np.float32)]
    ).astype(np.float32)
    grid.point_data['velocity'] = vel
    grid.point_data['obstacle'] = obs_flat

    # 保存为 VTK XML（ImageData，.vti）
    fname = os.path.join(out_dir, f'step_{step:05d}.vti')
    grid.save(fname)
                
def main():
    # Minimal GUI loop to render density (1) and velocity magnitude (2)
    initialize()
    gui = ti.GUI("LBM D2Q9", res=(Nx, Ny))
    cmap = cm.get_cmap(cmap_name)
    # Prepare ./simufigs: clear then recreate before simulation
    out_dir = './simufigs'
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    # Prepare ./vtk for ParaView
    vtk_out_dir = './vtk'
    shutil.rmtree(vtk_out_dir, ignore_errors=True)
    os.makedirs(vtk_out_dir, exist_ok=True)
    img_mode = 1  # 0: density, 1: |u|
    it = 0
    render_skip = 50  # render every 'skip' steps
    vtk_skip = 50  # export VTK every 'vtk_skip' steps
    max_steps = 20000  # maximum simulation steps
    frame = 0
    vtk_frame = 0
    pbar = tqdm(total=max_steps, desc="Simulating", unit="step")
    while gui.running and it < max_steps:
        # Simulation step
        copy_botzf()
        streaming()
        collision()
        boundary_condition()
        # Check for NaN values - stop if any detected
        if check_nan() > 0:
            print(f"NaN detected in botzf at step {it}. Simulation stopped.")
            break

        # Render
        if it % render_skip == 0:
            paint(img_mode)
            gui.set_image(cmap(img.to_numpy()))
            gui.show(os.path.join(out_dir, f'frame_{frame:05d}.png'))
            frame += 1
        # Export VTK for ParaView
        if it % vtk_skip == 0:
            write_vtk(vtk_frame, vtk_out_dir)
            vtk_frame += 1
        it += 1
        pbar.update(1)

        # Simple key toggles: '1' for density, '2' for velocity magnitude
        if gui.get_event():
            if gui.event.key == '1':
                img_mode = 0
            elif gui.event.key == '2':
                img_mode = 1
    pbar.close()

# def main():
#     initialize()
#     collision()
#     streaming()
#     boundary_condition()

if __name__ == '__main__':
    main()