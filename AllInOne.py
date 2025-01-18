import matplotlib.pyplot as plt
from math import sin, cos, tan
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

import netCDF4

###############################################################################
# 1) 读取并准备全球洋流数据
###############################################################################
filename = "oscar_currents_interim_20200101.nc"
ds = netCDF4.Dataset(filename, mode='r')

lat_arr = ds.variables["lat"][:]   # (nLat,)
lon_arr = ds.variables["lon"][:]   # (nLon,)
u_3d = ds.variables["u"]          # shape: (time, nLon, nLat)
v_3d = ds.variables["v"]          # shape: (time, nLon, nLat)

time_index = 0  # 若有多时刻可自行插值

def nearest_uv(lat, lon):
    """在(lat_arr, lon_arr)网格上做最近点查找, 返回(u, v)."""
    lon_idx = np.abs(lon_arr - lon).argmin()
    lat_idx = np.abs(lat_arr - lat).argmin()

    u_val = u_3d[time_index, lon_idx, lat_idx]
    v_val = v_3d[time_index, lon_idx, lat_idx]
    # 如果是 masked 值, 用 np.nan 填充
    u_filled = np.ma.filled(u_val, np.nan)
    v_filled = np.ma.filled(v_val, np.nan)
    return float(u_filled), float(v_filled)

###############################################################################
# 2) 6-DOF 仿真辅助
###############################################################################
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt/2, y + dt*k1/2)
    k3 = func(t + dt/2, y + dt*k2/2)
    k4 = func(t + dt,   y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# 目标本体 & 环境参数 (数值均视为在'度'或无物理量纲下，仅演示)
m    = 30000.0        # 质量(kg) (暂保留, 不从度换算)
rho  = 10250.0        # 海水密度(kg/m^3) (暂保留)
Vol  = 1e-4          # 用“度^3”? 这里仅演示, 没有物理意义
L    = 0.00009          # 长度 ~ 0.1度 (相当于~10~11公里)
D    = 0.00005          # 直径 ~ 0.05度
g    = 0.00000981          # 重力加速度(m/s^2) (仍是物理常量, 未改)

# 阻力系数(默认)
Cd_x = 1.0
Cd_y = 1.1
Cd_z = 1.2

# 时间仍用秒为单位, dt=10 s, t_end=100 s
dt    = 0.01
t_end = 5.0

# 初始状态 (单位: X, Y 以“度”, Z 以“度”...仅演示)
X0 = 19.0        # 经度
Y0 = 37.0        # 纬度
Z0 = -0.02       # “深度”, 这里随意改小数度, 仅演示
phi0   = 0.0
theta0 = 0.0
psi0   = 0.0
U10    = 0.0
U20    = 0.0
U30    = 0.0
Om10   = 0.0
Om20   = 0.0
Om30   = 0.0

initial_state = np.array([
    X0, Y0, Z0,
    phi0, theta0, psi0,
    U10, U20, U30,
    Om10, Om20, Om30
])

wave_amp   = 0.5
wave_omega = 2.0

def rotation_matrix_body_to_inertial(phi, theta, psi):
    Rz = np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi),  cos(psi), 0],
        [0,         0,        1]
    ])
    Ry = np.array([
        [ cos(theta), 0, sin(theta)],
        [ 0,          1, 0],
        [-sin(theta), 0, cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi),  cos(phi)]
    ])
    return Rz @ Ry @ Rx

def wave_current_velocity(t, X, Y, Z, phi, theta, psi):
    """用最近点插值得到(u,v), 叠加简化波浪."""
    # 1) 洋流
    Vc_x, Vc_y = nearest_uv(Y, X)  # 注意: X->lon, Y->lat
    Vc_z = 0.0

    # 2) 波浪
    Vw_x = wave_amp*wave_omega*cos(wave_omega*t)
    Vw_y = 0.0
    Vw_z = wave_amp*wave_omega*sin(wave_omega*t)

    return (Vc_x + Vw_x,
            Vc_y + Vw_y,
            Vc_z + Vw_z)

def compute_forces_and_moments(t, state):
    X, Y, Z, phi, theta, psi, U1, U2, U3, Om1, Om2, Om3 = state

    R_body2inertial = rotation_matrix_body_to_inertial(phi, theta, psi)
    vel_inertial    = R_body2inertial @ np.array([U1, U2, U3])
    X_dot, Y_dot, Z_dot = vel_inertial

    phi_dot   = Om1
    theta_dot = Om2
    psi_dot   = Om3

    # 洋流 + 波浪(度/s)
    Vcw_x, Vcw_y, Vcw_z = wave_current_velocity(t, X, Y, Z, phi, theta, psi)

    # 相对速度(度/s)
    rel_vel_inertial = np.array([Vcw_x, Vcw_y, Vcw_z]) - vel_inertial
    R_inertial2body  = R_body2inertial.T
    Ux_rel, Uy_rel, Uz_rel = R_inertial2body @ rel_vel_inertial

    # 力(非常不具物理意义, 因为把度当长度来算)
    Fg_net = (m - rho*Vol)*g
    Fx_g =  Fg_net*sin(theta)
    Fy_g = -Fg_net*cos(theta)*sin(phi)
    Fz_g = -Fg_net*cos(theta)*cos(phi)

    def sign(v):
        return 1.0 if v>=0 else -1.0

    # 以“度^2”来算截面积…
    area_x   = np.pi*(D**2)/4
    area_side= D*L

    Fx_d = 0.5*rho*Cd_x*area_x*(Ux_rel**2)*sign(Ux_rel)
    Fy_d = 0.5*rho*Cd_y*area_side*(Uy_rel**2)*sign(Uy_rel)
    Fz_d = 0.5*rho*Cd_z*area_side*(Uz_rel**2)*sign(Uz_rel)

    Fx_body = Fx_g + Fx_d
    Fy_body = Fy_g + Fy_d
    Fz_body = Fz_g + Fz_d

    M_eff = m - rho*Vol
    U1_dot = (Fx_body - M_eff*(U3*Om2 - U2*Om3))/M_eff
    U2_dot = (Fy_body - M_eff*(U1*Om3 - U3*Om1))/M_eff
    U3_dot = (Fz_body - M_eff*(U2*Om1 - U1*Om2))/M_eff

    Ixx, Iyy, Izz = 0.1, 0.2, 0.2
    Mx = 0.0
    My = -0.01*Om2
    Mz = -0.01*Om3

    Om1_dot = Mx/Ixx
    Om2_dot = My/Iyy
    Om3_dot = Mz/Izz

    return np.array([
        X_dot, Y_dot, Z_dot,
        phi_dot, theta_dot, psi_dot,
        U1_dot, U2_dot, U3_dot,
        Om1_dot, Om2_dot, Om3_dot
    ])

def simulate_once(initial_state, dt, t_end):
    N_steps = int(t_end/dt) + 1
    states  = np.zeros((N_steps, len(initial_state)))
    y = initial_state.copy()
    for i in range(N_steps):
        states[i,:] = y
        y = rk4_step(compute_forces_and_moments, i*dt, y, dt)
    return states


###############################################################################
# 4) Monte-Carlo
###############################################################################
num_samples = 10
results_all = []

for iMC in range(num_samples):
    # 可随机阻力或初始速度等
    new_init = initial_state.copy()
    new_init[6] = random.uniform(-0.1, 0.1)  # U1
    new_init[9] = random.uniform(-0.05, 0.05)# Om1

    states_arr = simulate_once(new_init, dt, t_end)
    results_all.append(states_arr)

X_end, Y_end, Z_end = [], [], []
for arr in results_all:
    X_end.append(arr[-1,0])
    Y_end.append(arr[-1,1])
    Z_end.append(arr[-1,2])


# =====================================================================
# 5. 可视化（2D）
# =====================================================================
fig = plt.figure(figsize=(10, 6))
ax_xy = fig.add_subplot(1, 2, 1)
ax_xz = fig.add_subplot(1, 2, 2)

# 首先收集落点信息
X_all_end = []
Y_all_end = []
Z_all_end = []

# 先画多条轨迹 + 收集最终落点
for i, res in enumerate(results_all):
    X_ = res[:, 0]
    Y_ = res[:, 1]
    Z_ = res[:, 2]
    ax_xy.plot(X_, Y_, '-', alpha=0.5)
    ax_xz.plot(X_, Z_, '-', alpha=0.5)

    X_all_end.append(X_[-1])
    Y_all_end.append(Y_[-1])
    Z_all_end.append(Z_[-1])

# 在 XY 和 XZ 平面上画落点散点
ax_xy.scatter(X_all_end, Y_all_end, c='red', marker='o', alpha=0.6, label='Landing points')
ax_xz.scatter(X_all_end, Z_all_end, c='red', marker='o', alpha=0.6)

ax_xy.set_xlabel('X (m)')
ax_xy.set_ylabel('Y (m)')
ax_xy.set_title('XY-plane Trajectories')
ax_xy.legend()
ax_xy.grid(True)

ax_xz.set_xlabel('X (m)')
ax_xz.set_ylabel('Z (m)')
ax_xz.set_title('XZ-plane Trajectories')
ax_xz.grid(True)

plt.tight_layout()
plt.show()


# =====================================================================
# 6. 可视化（3D + 包围盒）
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

##########################################
# 1. 收集并统计所有轨迹的 (X, Y, Z)
##########################################
all_X = []
all_Y = []
all_Z = []

# 假设 results_all 中每条轨迹 shape=(N_steps,12)，其中 [:,0],[: ,1],[: ,2] 分别为 X, Y, Z
for res in results_all:
    all_X.extend(res[:,0])
    all_Y.extend(res[:,1])
    all_Z.extend(res[:,2])

# 转为 numpy 数组，方便做 min / max 等计算
all_X = np.array(all_X)
all_Y = np.array(all_Y)
all_Z = np.array(all_Z)

# 这里也可以直接重用之前的 X_all_end, Y_all_end, Z_all_end
# 若你不在上面收集过，也可这样再收集:
# X_all_end = [res[-1,0] for res in results_all]
# Y_all_end = [res[-1,1] for res in results_all]
# Z_all_end = [res[-1,2] for res in results_all]

##########################################
# 2. 绘制 3D 多条轨迹 + 最终落点
##########################################
fig = plt.figure(figsize=(10,8))
ax_3d = fig.add_subplot(111, projection='3d')

# 2.1 画 Monte-Carlo 多条 3D 轨迹
for i, res in enumerate(results_all):
    X_ = res[:,0]
    Y_ = res[:,1]
    Z_ = res[:,2]
    ax_3d.plot(X_, Y_, Z_, alpha=0.6)

# 2.2 画最终落点散点
ax_3d.scatter(X_all_end, Y_all_end, Z_all_end, c='red', marker='o',
              alpha=0.8, label='Landing Points')

# 设置坐标轴与标题
ax_3d.set_xlabel('X (m)')
ax_3d.set_ylabel('Y (m)')
ax_3d.set_zlabel('Z (m)')
ax_3d.set_title('3D Trajectories of the Cylinder')
ax_3d.legend()
ax_3d.grid(True)

##########################################
# 3. 绘制 3D 包围盒 (Bounding Box)
##########################################
# 3.1 计算全局最小最大值
X_min, X_max = np.min(all_X), np.max(all_X)
Y_min, Y_max = np.min(all_Y), np.max(all_Y)
Z_min, Z_max = np.min(all_Z), np.max(all_Z)

def draw_bounding_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, **kwargs):
    """
    在 ax 上绘制包围盒(立方体)的 12 条边。
    kwargs 可以传递给 ax.plot()，如 color, linestyle 等。
    """
    # 8 个顶点
    corners = [
        (xmin, ymin, zmin),
        (xmin, ymin, zmax),
        (xmin, ymax, zmin),
        (xmin, ymax, zmax),
        (xmax, ymin, zmin),
        (xmax, ymin, zmax),
        (xmax, ymax, zmin),
        (xmax, ymax, zmax),
    ]
    # 每条边由顶点索引组成
    edges = [
        (0,1), (0,2), (0,4),
        (3,1), (3,2), (3,7),
        (5,1), (5,4), (5,7),
        (6,2), (6,4), (6,7),
    ]
    # 逐条绘制
    for (i1, i2) in edges:
        xvals = [corners[i1][0], corners[i2][0]]
        yvals = [corners[i1][1], corners[i2][1]]
        zvals = [corners[i1][2], corners[i2][2]]
        ax.plot(xvals, yvals, zvals, **kwargs)

# 在 3D 坐标系中画包围盒
draw_bounding_box(ax_3d, X_min, X_max, Y_min, Y_max, Z_min, Z_max,
                  color='magenta', linestyle='--', linewidth=1.5)

# 若想给包围盒做个图例，可在这里额外画一个不可见线对象：
ax_3d.plot([], [], [], color='magenta', linestyle='--', label='Bounding Box')
ax_3d.legend()

##########################################
# 5. 显示图像
##########################################
plt.tight_layout()
plt.show()

