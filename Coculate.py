import matplotlib.pyplot as plt
from math import sin, cos, tan, sqrt
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

# 若需要更准确/通用的数值积分方法，可以使用 scipy.integrate.ode 或 solve_ivp。
# 这里示例使用自定义的RK4函数。
def rk4_step(func, t, y, dt, **kwargs):
    """
    4阶Runge-Kutta单步，用于积分 dy/dt = func(t, y).
    y: 状态变量 (numpy array)
    dt: 步长
    func(t, y): 返回 dy/dt (numpy array)
    """
    k1 = func(t, y, **kwargs)
    k2 = func(t + dt / 2, y + dt * k1 / 2, **kwargs)
    k3 = func(t + dt / 2, y + dt * k2 / 2, **kwargs)
    k4 = func(t + dt, y + dt * k3, **kwargs)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# =====================================================================
# 1. 参数配置 (示例取值, 用户可自行调整)
# =====================================================================
# 目标本体参数 (以圆柱为例)
m = 1000.0  # 目标质量(kg)
rho = 1025.0  # 海水密度(kg/m^3)
Vol = 0.00977  # 排水体积( ~ 体积 ), 这里仅做示例
L = 0.45  # 圆柱长度(m)
D = 0.01  # 圆柱直径(m)
g = 9.81  # 重力加速度(m/s^2)
nu = 1e-6  # 流体运动黏度(仅示例)

# 阻力系数(可设为随机范围, 此示例先固定)
Cd_x = 1.0
Cd_y = 1.1
Cd_z = 1.2

# 假设附加质量简化为常数 (可根据文献/经验给出)
m22 = 1.0
m33 = 2.0
m55 = 0.1
m66 = 0.1

# x_t 偏置 (示例)
x_t = 0.0

# 环境参数 (简化示例：波浪频率、振幅、洋流等)
wave_amp = 0.5  # A
wave_omega = 2.0  # ω
wave_dir = 0.0  # α, 与 X 轴平行
current_vx = 0.05  # 洋流 X 方向速度
current_vy = 0.0  # 洋流 Y 方向速度
current_vz = 0.0  # 洋流 Z 方向速度

# 数值积分相关
dt = 1  # 时间步长
t_end = 50.0  # 模拟总时长

# 初始条件 (位置, 姿态, 速度, 角速度)
# y = [X, Y, Z, phi, theta, psi, U1, U2, U3, Omega1, Omega2, Omega3]
# 其中 X, Y, Z 和 phi,theta,psi 在地理坐标系; U1,U2,U3 在体轴坐标系(如文中),
# Omega1,Omega2,Omega3 在地理坐标系(或自行统一到体轴坐标系, 需对应修改方程).
X0 = 0.0
Y0 = 0.0
Z0 = 0.0
phi0 = 0.0
theta0 = 0.0
psi0 = 0.0
U10 = 0.0
U20 = 0.0
U30 = 0.0
Om10 = 0.0
Om20 = 0.0
Om30 = 0.0

initial_state = np.array([X0, Y0, Z0,
                          phi0, theta0, psi0,
                          U10, U20, U30,
                          Om10, Om20, Om30])


# =====================================================================
# 2. 辅助函数：姿态转换、力与力矩计算等
# =====================================================================
def euler_to_omega(phi, theta, psi, dphi, dtheta, dpsi):
    """
    计算地理坐标系下的角速度 (Omega1, Omega2, Omega3)，
    对应文中(7)式的逆变换或正变换。
    注：下面是将 (dphi,dtheta,dpsi) -> (Omega1,Omega2,Omega3) 的直接矩阵乘法:
        [Omega1]   [ 1      sinφ tanθ       cosφ tanθ   ] [ dphi   ]
        [Omega2] = [ 0      cosφ            -sinφ       ] [ dtheta ]
        [Omega3]   [ 0      sinφ /cosθ      cosφ /cosθ  ] [ dpsi   ]
    """
    # 这里示例用显式写法，如果直接用矩阵，可以 np.dot(...)。
    Omega1 = dphi + sin(phi) * tan(theta) * dtheta + cos(phi) * tan(theta) * dpsi
    Omega2 = 0.0 + cos(phi) * dtheta - sin(phi) * dpsi
    Omega3 = 0.0 + sin(phi) / cos(theta) * dtheta + cos(phi) / cos(theta) * dpsi
    return np.array([Omega1, Omega2, Omega3])


def wave_current_velocity(t, X, Y, Z, phi, theta, psi):
    """
    计算波浪+洋流耦合后的流场速度在 地理坐标系 下的分量 (Vcx, Vcy, Vcz)，
    以及波浪频率等对水深的影响(如 Airy 波 e^{kZ}, 此处仅做简单示例)。
    """
    # 简化: Vw_x = A*omega * cos(omega*t)  (假设波向=0°, 不考虑空间变化)
    #       Vw_y = 0
    #       Vw_z = A*omega * sin(omega*t)
    # 真实情况请根据文献公式(20)和(22)进行坐标变换
    Vw_x = wave_amp * wave_omega * cos(wave_omega * t)
    Vw_y = 0.0
    Vw_z = wave_amp * wave_omega * sin(wave_omega * t)

    # 洋流
    Vc_x = current_vx
    Vc_y = current_vy
    Vc_z = current_vz

    # 总流速(地理系)
    Vcw_x = Vc_x + Vw_x
    Vcw_y = Vc_y + Vw_y
    Vcw_z = Vc_z + Vw_z

    return Vcw_x, Vcw_y, Vcw_z


def rotation_matrix_body_to_inertial(phi, theta, psi):
    """
    计算 R = R_z(psi)*R_y(theta)*R_x(phi),
    或文中定义的 body->inertial 的旋转矩阵(也可能是 inertial->body 的逆)。
    这里请确保和速度/方程对应时用对了矩阵正/逆方向。
    """
    # 此处仅示例，如要严格一致需看文中(7)(22)(27)的约定方向
    Rz = np.array([[cos(psi), -sin(psi), 0],
                   [sin(psi), cos(psi), 0],
                   [0, 0, 1]])
    Ry = np.array([[cos(theta), 0, sin(theta)],
                   [0, 1, 0],
                   [-sin(theta), 0, cos(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, cos(phi), -sin(phi)],
                   [0, sin(phi), cos(phi)]])
    R = Rz @ Ry @ Rx
    return R


def compute_forces_and_moments(t, state):
    """
    根据当前 state 计算6个自由度的力、力矩(或加速度)。
    返回: d(state)/dt = [ dX/dt, dY/dt, dZ/dt, dphi/dt, dtheta/dt, dpsi/dt,
                         dU1/dt, dU2/dt, dU3/dt, dOm1/dt, dOm2/dt, dOm3/dt ]
    其中前6个主要是 kinematic, 后6个是动力学。
    """
    X, Y, Z, phi, theta, psi, U1, U2, U3, Om1, Om2, Om3 = state

    # 1) 位置和姿态的导数(几何关系)
    #    X_dot, Y_dot, Z_dot 由 速度(U1,U2,U3) 在地理系投影得到。
    #    若文中(27)为 body->inertial，则 [X_dot, Y_dot, Z_dot]^T = R(phi,theta,psi)*[U1, U2, U3]^T
    R_body2inertial = rotation_matrix_body_to_inertial(phi, theta, psi)
    vel_inertial = R_body2inertial @ np.array([U1, U2, U3])
    X_dot, Y_dot, Z_dot = vel_inertial

    # 2) 姿态角的导数(若使用欧拉角，需由(Om1,Om2,Om3)倒推 dphi,dtheta,dpsi)
    #    见文中(7)，我们这里简单假设 Om1= dphi, Om2= dtheta, Om3=dpsi 的近似(不够严谨),
    #    或写一个逆矩阵 E_inv * [Om1,Om2,Om3]^T = [dphi,dtheta,dpsi]^T
    #    这里为演示，做个简化，假设绕X,Y,Z的小角度 => dphi = Om1, dtheta=Om2, dpsi=Om3
    #    (请在实际中替换为准确矩阵)
    phi_dot = Om1
    theta_dot = Om2
    psi_dot = Om3

    # 3) 计算体坐标系下的相对速度(流场 - 刚体速度)
    #    在地理坐标系下耦合流速:
    Vcw_x, Vcw_y, Vcw_z = wave_current_velocity(t, X, Y, Z, phi, theta, psi)
    #    将刚体速度(在体坐标U1,U2,U3) 转到地理坐标
    body_vel_inertial = vel_inertial  # 同X_dot, Y_dot, Z_dot
    rel_vel_inertial = np.array([Vcw_x, Vcw_y, Vcw_z]) - body_vel_inertial
    #    转回体坐标下:
    R_inertial2body = R_body2inertial.T  # 旋转矩阵的逆
    rel_vel_body = R_inertial2body @ rel_vel_inertial
    Ux_rel, Uy_rel, Uz_rel = rel_vel_body

    # 4) 计算各项力 (示例中极度简化，仅演示形式)
    #    - 重力浮力项: (m - rho*Vol)*g, 但要分解到体坐标 => 又需要姿态
    #    - 阻力:
    #       Fx_drag ~ 1/2*rho*Cd_x*A*(Ux_rel^2)*sign(Ux_rel) + 0.664*...
    #    - 类似地, Fy_drag, Fz_drag
    #    - 升力、黏性力矩 (此处不展开积分形式, 只做示例)

    W = m * g  # 重力
    B = rho * Vol * g  # 浮力
    # 重浮力净值:
    Fg_net = (m - rho * Vol) * g
    # 在体坐标系下, 由于theta, phi, etc.会有分解:
    # 简化：假设theta为俯仰, phi为横摇:
    # X方向(体坐标)的重力分量 ~ Fg_net*sin(theta)
    # Y方向(体坐标)的重力分量 ~ -Fg_net*cos(theta)*sin(phi)
    # Z方向(体坐标)的重力分量 ~ -Fg_net*cos(theta)*cos(phi)
    Fx_g = Fg_net * sin(theta)
    Fy_g = -Fg_net * cos(theta) * sin(phi)
    Fz_g = -Fg_net * cos(theta) * cos(phi)

    # 阻力(仅演示对 Ux_rel, Uy_rel, Uz_rel 做二次方项)
    # 注意文中(8)中还包含 0.664 sqrt(...) 的边界层阻力，用户可自行补充。
    def sign(v):
        return 1.0 if v >= 0 else -1.0

    area_x = np.pi * (D ** 2) / 4.0  # 截面积(简化)
    # Fx_drag
    Fx_d = 0.5 * rho * Cd_x * area_x * (Ux_rel ** 2) * sign(Ux_rel)
    # 侧向阻力(假设圆柱侧面积 L*D, 做近似)
    area_side = D * L
    Fy_d = 0.5 * rho * Cd_y * area_side * (Uy_rel ** 2) * sign(Uy_rel)
    Fz_d = 0.5 * rho * Cd_z * area_side * (Uz_rel ** 2) * sign(Uz_rel)

    # 总体坐标系下的合力 (不含其他升力、涡激力等):
    Fx_body = Fx_g + Fx_d
    Fy_body = Fy_g + Fy_d
    Fz_body = Fz_g + Fz_d

    # 5) 平动方程 => 质量*(dU1/dt + 交叉项) = Fx_body
    #    交叉项: U3*Om2 - U2*Om3 (如文中(1)–(3))
    #    注意: 若要加上附加质量 (m - rho*Vol?), 或 m_{22}, m_{33}，可在左侧修正
    #    这里演示, 仅用“有效质量 Mx = m - rho*Vol”
    M_eff = m - rho * Vol
    # dxdt_U1
    U1_dot = (Fx_body - M_eff * (U3 * Om2 - U2 * Om3)) / M_eff
    U2_dot = (Fy_body - M_eff * (U1 * Om3 - U3 * Om1)) / M_eff
    U3_dot = (Fz_body - M_eff * (U2 * Om1 - U1 * Om2)) / M_eff

    # 6) 转动方程 => I*(dOm1/dt) = M 等(示例大幅简化)
    #    这里随意指定 Ixx, Iyy, Izz 近似, 或用 m55, m66...
    Ixx = 0.1
    Iyy = 0.2
    Izz = 0.2
    # 这里不展开黏性力矩 M_{dy}, M_{dz}、升力矩 M_{Ly}, M_{Lz} 等的积分,
    # 仅示意生成小阻尼项:
    Mx = 0.0  # Om1_dot = 0 (参考文中(4)似乎假设常数?)
    My = -0.01 * Om2  # 假设与 Om2 成比例
    Mz = -0.01 * Om3  # 同理

    Om1_dot = Mx / Ixx
    Om2_dot = My / Iyy
    Om3_dot = Mz / Izz

    # 7) 拼装结果
    dydt = np.array([
        X_dot, Y_dot, Z_dot,  # dX/dt, dY/dt, dZ/dt
        phi_dot, theta_dot, psi_dot,  # dphi/dt, ...
        U1_dot, U2_dot, U3_dot,  # dU1/dt, ...
        Om1_dot, Om2_dot, Om3_dot  # dOm1/dt, ...
    ])
    return dydt


# =====================================================================
# 3. 单次仿真函数
# =====================================================================
def simulate_once(initial_state, dt, t_end):
    """
    返回 (time_array, state_history)
    state_history.shape = (N_steps, 12)
    """
    N_steps = int(t_end / dt) + 1
    states = np.zeros((N_steps, len(initial_state)))
    times = np.zeros(N_steps)
    y = initial_state.copy()
    for i in range(N_steps):
        states[i, :] = y
        times[i] = i * dt
        # RK4 单步
        y = rk4_step(compute_forces_and_moments, times[i], y, dt)
    return times, states


# =====================================================================
# 4. Monte-Carlo 多次试验 (示例)
# =====================================================================
num_samples = 20  # 可以自行加大以得到更平滑的统计分布

results_all = []
for iMC in range(num_samples):
    # 在此可以对初始条件或阻力系数、x_t 等进行随机化:
    #   1) 阻力系数
    Cd_x_rand = random.uniform(0.9, 1.1)
    Cd_y_rand = random.uniform(1.0, 1.2)
    Cd_z_rand = random.uniform(1.1, 1.3)
    #   2) 入水时角度 / 速度扰动
    Om10_rand = random.uniform(-0.05, 0.05)
    U10_rand = random.uniform(-0.1, 0.1)
    # 其他随机化...

    # 这里简单赋给 global (不严谨)
    Cd_x = Cd_x_rand
    Cd_y = Cd_y_rand
    Cd_z = Cd_z_rand

    # 构造新的初始状态
    state_rand = initial_state.copy()
    state_rand[6] = U10_rand + 1  # U1
    state_rand[9] = Om10_rand  # Om1

    # 做一次仿真
    t_arr, states_arr = simulate_once(state_rand, dt, t_end)

    results_all.append(states_arr)

# 生成轨迹包络: 例如只关心 (X(t_end), Y(t_end), Z(t_end))
# 或者每个时间步的 min/max
X_all_end = []
Y_all_end = []
Z_all_end = []
for res in results_all:
    X_all_end.append(res[-1, 0])
    Y_all_end.append(res[-1, 1])
    Z_all_end.append(res[-1, 2])

# =====================================================================
# 5. 可视化
# =====================================================================
fig = plt.figure(figsize=(10, 6))
ax_xy = fig.add_subplot(1, 2, 1)
ax_xz = fig.add_subplot(1, 2, 2)

# 先画多条轨迹
for i, res in enumerate(results_all):
    X_ = res[:, 0]
    Y_ = res[:, 1]
    Z_ = res[:, 2]
    ax_xy.plot(X_, Y_, '-', alpha=0.5)
    ax_xz.plot(X_, Z_, '-', alpha=0.5)

# 落点散点
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
# 6. 可视化
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

# 假设 results_all 中每条轨迹 shape=(N_steps,12)，其中[:,0],[:,1],[:,2] 分别为 X, Y, Z
for res in results_all:
    all_X.extend(res[:,0])
    all_Y.extend(res[:,1])
    all_Z.extend(res[:,2])

# 转为 numpy 数组，方便计算
all_X = np.array(all_X)
all_Y = np.array(all_Y)
all_Z = np.array(all_Z)

# 求出最终落点 (X_end, Y_end, Z_end)
X_all_end = [res[-1,0] for res in results_all]
Y_all_end = [res[-1,1] for res in results_all]
Z_all_end = [res[-1,2] for res in results_all]

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

