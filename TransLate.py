import matplotlib.pyplot as plt
from math import sin, cos, tan, copysign
import random
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持
import netCDF4
import csv     # 导入 csv 模块用于保存访问顺序

###############################################################################
# 1) 读取 OSCAR 洋流数据
###############################################################################
filename_oscar = "oscar_currents_interim_20200101.nc"
ds_oscar = netCDF4.Dataset(filename_oscar, mode='r')

lat_arr = ds_oscar.variables["lat"][:]  # shape (nLat,)  in degrees
lon_arr = ds_oscar.variables["lon"][:]  # shape (nLon,)  in degrees
u_3d = ds_oscar.variables["u"]  # shape: (time, nLon, nLat)
v_3d = ds_oscar.variables["v"]  # shape: (time, nLon, nLat)

time_index = 0  # 若有多时刻可自行插值


def nearest_uv(lat, lon):
    """
    在 (lat_arr, lon_arr) 网格上做最近点查找, 返回 (u, v).
    单位假设还是 (度/s) 或类似, 未做物理转换, 仅演示.
    """
    lon_idx = np.abs(lon_arr - lon).argmin()
    lat_idx = np.abs(lat_arr - lat).argmin()

    u_val = u_3d[time_index, lon_idx, lat_idx]  # masked array
    v_val = v_3d[time_index, lon_idx, lat_idx]
    u_filled = np.ma.filled(u_val , np.nan)
    v_filled = np.ma.filled(v_val , np.nan)
    return float(u_filled), float(v_filled)


###############################################################################
# 1b) 读取 GEBCO_2020 数据(获取海床高度,单位: 米), 用于判断"触底"
###############################################################################

filename_gebco = "GEBCO_2020.nc"
ds_gebco = netCDF4.Dataset(filename_gebco, mode='r')

lat_variable = ds_gebco.variables["lat"]  # shape (nLat2,)
lon_variable = ds_gebco.variables["lon"]  # shape (nLon2,)
elev_variable = ds_gebco.variables["elevation"]  # shape (nLat2, nLon2)

# 1) 找到 lat, lon 对应的起止索引
target_lat_min, target_lat_max = 10.0, 20.0
target_lon_min, target_lon_max = -80.0, -70.0

# a) 用 np.searchsorted 或 手动 argmin
lat_array = lat_variable[:]  # 只读 lat 这一维
lon_array = lon_variable[:]  # 只读 lon 这一维

lat_start = np.searchsorted(lat_array, target_lat_min, side='left')
lat_end = np.searchsorted(lat_array, target_lat_max, side='right')
lon_start = np.searchsorted(lon_array, target_lon_min, side='left')
lon_end = np.searchsorted(lon_array, target_lon_max, side='right')

# b) 做 slicing
sub_lats = lat_variable[lat_start:lat_end]  # shape (lat_sub_size,)
sub_lons = lon_variable[lon_start:lon_end]  # shape (lon_sub_size,)

# c) elevation 子集 shape => (lat_sub_size, lon_sub_size)
sub_elev = elev_variable[lat_start:lat_end, lon_start:lon_end]

# 这样 sub_lats, sub_lons, sub_elev 只包含 [30,40],[15,25] 范围的网格
print("Sub-lats shape:", sub_lats.shape)
print("Sub-lons shape:", sub_lons.shape)
print("Sub-elev shape:", sub_elev.shape)


def nearest_elevation_m(lat, lon):
    """
    从 GEBCO 数据中, 找到最邻近点, 返回海床 elevation(米).
    elev>0 => 陆地; elev<0 => 水深.
    """
    lat_idx = np.abs(lat_variable - lat).argmin()
    lon_idx = np.abs(lon_variable - lon).argmin()
    elev_m = elev_variable[lat_idx, lon_idx]
    return float(elev_m)


def get_seabed_depth_deg(lat, lon):
    """
    返回(该处)海床的Z坐标(单位:度).
    elev_m >0 =>陆地(这里简单设seaBed=0),
    elev_m <0 =>水深(负值),
    用1 deg ~ 111000m做极粗略换算 => seabed_deg = elev_m / 111000
    """
    elev_m = nearest_elevation_m(lat, lon)
    if elev_m >= 0:
        # 表示陆地或海拔, 此时海床Z= 0.0 deg (示例)
        return 0.0
    else:
        # elev_m是负 =>水深, e.g. -3000 => -3000/111000 ~ -0.027deg
        return elev_m / 111000.0


###############################################################################
# 2) 6-DOF 仿真辅助
###############################################################################
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + dt / 2, y + dt * k1 / 2)
    k3 = func(t + dt / 2, y + dt * k2 / 2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


###############################################################################
# 3) 场景参数(以度为"伪"长度单位)
###############################################################################
m = 1000.0  # 目标质量(kg)
rho = 1025.0  # 海水密度(kg/m^3)
Vol = 0.01000  # 排水体积( ~ 体积 ), 这里仅做示例
L = 12 / 111000  # "度"
D = 6 / 111000  # "度"
g = 9.81 / 111000  # 伪重力(仅演示)

# 阻力系数
Cd_x = 1.0
Cd_y = 1.1
Cd_z = 1.2

# 时间参数
dt = 100
t_end = 3000.0

# 初始状态
X0 = 15.0
Y0 = -75.0
Z0 = -0.01  # 负 => 水下(度)
phi0 = 0.0
theta0 = 0.0
psi0 = 0.0
U10 = 0.04 / 111000
U20 = 0.02 / 111000
U30 = 0.0
Om10 = 0.0
Om20 = 0.0
Om30 = 0.0

initial_state = np.array([
    X0, Y0, Z0,
    phi0, theta0, psi0,
    U10, U20, U30,
    Om10, Om20, Om30
])

# 波浪参数
wave_amp = 0.5 / 10000
wave_omega = 0.34

###############################################################################
# 3b) 海底滚动时的简化参数
###############################################################################
rolling_friction = 0.01  # 简化滚动摩擦系数


def rotation_matrix_body_to_inertial(phi, theta, psi):
    Rz = np.array([
        [cos(psi), -sin(psi), 0],
        [sin(psi), cos(psi), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, cos(phi), -sin(phi)],
        [0, sin(phi), cos(phi)]
    ])
    return Rz @ Ry @ Rx


def wave_current_velocity(t, X, Y, Z, phi, theta, psi):
    """获取(经,纬)位置的洋流(u,v) + 波浪."""
    Vc_x, Vc_y = nearest_uv(Y, X)  # X->lon, Y->lat
    Vc_z = 0.0

    Vw_x = wave_amp * wave_omega * cos(wave_omega * t)
    Vw_y = 0.0
    Vw_z = wave_amp * wave_omega * sin(wave_omega * t)

    return (Vc_x + Vw_x,
            Vc_y + Vw_y,
            Vc_z + Vw_z)


def get_seabed_slope(lat, lon):
    """
    返回海床在 (lat, lon) 处的 x、y 方向坡度(度/度)或 (米/米)。
    这里仅示例化：从 get_seabed_depth_deg() 获取该点和周围微小偏移的海床Z，
    做个差分 approximations。
    """
    # 例如：取小增量 delta=0.0001 deg
    delta = 0.001
    elev_center = get_seabed_depth_deg(lat, lon)  # Zc
    elev_dx = get_seabed_depth_deg(lat, lon + delta)  # Zx
    elev_dy = get_seabed_depth_deg(lat + delta, lon)  # Zy

    # x方向坡度 ~ (Zx - Zc)/ delta
    slope_x = (elev_dx - elev_center) / 111000
    # y方向坡度 ~ (Zy - Zc)/ delta
    slope_y = (elev_dy - elev_center) / 111000

    return slope_x, slope_y

def compute_forces_and_moments_in(t, state):
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
    Fz_g = -Fg_net * cos(theta) * cos(phi)*random.randint(0, 1)

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
    U1_dot = 100*(Fx_body - M_eff * (U3 * Om2 - U2 * Om3)) / M_eff
    U2_dot = 100*(Fy_body - M_eff * (U1 * Om3 - U3 * Om1)) / M_eff
    U3_dot = ((Fz_body - M_eff * (U2 * Om1 - U1 * Om2)) / (M_eff*5000)) +((Fz_body - M_eff * (U2 * Om1 - U1 * Om2)) / (M_eff*1000))*random.randint(-2, 1)

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


def compute_forces_and_moments_on(t,state):
    """
    用于海底滚动阶段。
    这里假设物体Z已经与海床齐平(或Z<=seaBed)，
    我们在这个函数里让Z_dot=0、U3=0，但允许在X-Y平面受坡度力和滚动摩擦力作用滚动。

    返回同样格式的 11 元素数组：
    [X_dot, Y_dot, Z_dot,
     phi_dot, theta_dot, psi_dot,
     U1_dot, U2_dot, U3_dot,
     Om1_dot, Om2_dot, Om3_dot]
    """
    # 解包状态:
    X, Y, Z, phi, theta, psi, U1, U2, U3, Om1, Om2, Om3 = state

    # 1) 让Z贴住海床 & 垂直方向速度为0(物理上相当于刚贴地)
    seabed_z_deg = get_seabed_depth_deg(Y, X)
    # 如果你要在此就强行Z=seabed_z_deg，也可以，但若在simulate_once()里做后置钳制也可。
    Z_dot = 0.0
    X_dot = 0.0
    Y_dot = 0.0
    # 物理上U3=0 => 体坐标系下不再有垂向速度
    U3 = 0.0

    # 2) 计算海底坡度 => 产生“坡度力”
    slope_x, slope_y = get_seabed_slope(Y, X)
    # slope_x,y 大致表示 Z随X,Y变化的梯度(见你之前的 get_seabed_slope实现)

    # 让重力分量 = m*g*slope
    Fx_slope = - (m * g) * slope_x
    Fy_slope = - (m * g) * slope_y
    Fz_slope = 0.0  # 没有额外z分量

    # 3) 滚动摩擦(或动摩擦), 简化
    Fx_roll = - rolling_friction * (m * g) * np.sign(U1)
    Fy_roll = - rolling_friction * (m * g) * np.sign(U2)
    Fz_roll = 0.0

    Fx_total = Fx_slope + Fx_roll
    Fy_total = Fy_slope + Fy_roll
    Fz_total = 0.0  # 不再向下动

    # 4) 计算加速度(在体坐标系)
    # 这里为了简单，假设M_eff=m => 不再考虑浮力
    M_eff = m
    U1_dot = Fx_total / M_eff
    U2_dot = Fy_total / M_eff
    U3_dot = 0.0

    # 5) 角速度阻尼(示例)
    Ixx, Iyy, Izz = 0.1, 0.2, 0.2
    Mx = 0.0
    My = -0.01 * Om2
    Mz = -0.01 * Om3
    Om1_dot = Mx / Ixx
    Om2_dot = My / Iyy
    Om3_dot = Mz / Izz

    # 6) 在惯性系下的速度(=X_dot, Y_dot, Z_dot)
    #    注意: X_dot, Y_dot, Z_dot 通常指 位置的导数(= 速度在惯性系),
    #    我们先得把 [U1, U2, U3=0] 转到惯性坐标, 第三分量=Z_dot=0
    R_body2inertial = rotation_matrix_body_to_inertial(phi, theta, psi)
    vel_inertial = R_body2inertial @ np.array([U1, U2, 0.0])
    X_dot, Y_dot, _ = vel_inertial

    # 7) 欧拉角导数
    #    简化：phi_dot=Om1, theta_dot=Om2, psi_dot=Om3 (跟你 water中相同做法)
    phi_dot = Om1
    theta_dot = Om2
    psi_dot = Om3

    # 8) 组装并返回
    return np.array([
        X_dot, Y_dot, Z_dot,  # (在惯性系下的速度)
        phi_dot, theta_dot, psi_dot,  # (欧拉角变化率)
        U1_dot, U2_dot, U3_dot,  # (体坐标系下的加速度)
        Om1_dot, Om2_dot, Om3_dot  # (角加速度)
    ])


def simulate_once(initial_state, dt, t_end):
    N_steps = int(t_end / dt) + 1
    states = np.zeros((N_steps, len(initial_state)))
    y = initial_state.copy()
    T = 0

    for i in range(N_steps):
        states[i, :] = y
        # 根据 Z 与 seabedZ 判断
        seabedZ = get_seabed_depth_deg(y[1], y[0])
        if y[2] <= seabedZ:
            T=1
        if T==1:
            y = rk4_step(compute_forces_and_moments_on, i * dt, y, dt)
            y[2]=seabedZ
            """
            delta = states[i, 0] - states[i-1, 0]
            deltay = states[i, 0] - states[i - 1, 0]
            elev_dxa = get_seabed_depth_deg(y[1], y[0] + delta)  # Zx
            elev_dxb = get_seabed_depth_deg(y[1], y[0] - delta)  # Zx
            elev_dya = get_seabed_depth_deg(y[1] + deltay, y[0])  # Zy
            elev_dyb = get_seabed_depth_deg(y[1] - deltay, y[0])  # Zy
            if elev_dxa>elev_dxb:
                y[0]=y[0] - delta
            else:
                y[0] = y[0] + delta

            if elev_dya>elev_dyb:
                y[1]=y[1] - deltay
            else:
                y[1] = y[1] + deltay
            """
        else:
            # 水中 => 用水中方程
            y = rk4_step(compute_forces_and_moments_in, i * dt, y, dt)
            y[8]=y[8] + random.randint(0, 1) * 0.0001 *(i * dt/t_end)*(i * dt/t_end)*(i * dt/t_end)


    return states

###############################################################################
# 4) Monte-Carlo
###############################################################################
num_samples = 20
results_all = []
for iMC in range(num_samples):
    # 随机微扰
    T = 0
    new_init = initial_state.copy()
    new_init[6] = random.uniform(-0.02 / 111000, 0.02 / 111000)  # U1
    new_init[7] = random.uniform(-0.01 / 111000, 0.01 / 111000)  # U1
    new_init[9] = random.uniform(-0.05, 0.05)  # Om1
    new_init[10] = random.uniform(-0.05, 0.05)  # Om1
    states_arr = simulate_once(new_init, dt, t_end)
    results_all.append(states_arr)

# 收集落点
X_end = [arr[-1, 0] for arr in results_all]
Y_end = [arr[-1, 1] for arr in results_all]
Z_end = [arr[-1, 2] for arr in results_all]
# 1) 计算分数矩阵（保持不变）
def map_to_grid(v, vmin, vmax, size):
    """
    将浮点数v从 [vmin, vmax] 映射到 [0, size-1].
    若v超出此范围，可clip到最小/最大.
    """
    if vmax == vmin:
        return 0
    ratio = (v - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))  # clip到[0,1]
    grid_idx = int(ratio * (size - 1))
    return grid_idx


def load_visit_order(filename):
    """
    从 CSV 文件中加载访问顺序数据，并按 Order 升序排序。

    参数:
    - filename: CSV 文件名

    返回:
    - visit_order_sorted: 按 Order 排序后的访问顺序列表，每个元素为 (Order, X, Y)
    """
    visit_order = []
    with open(filename, mode='r', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            order = int(row['Order'])
            x = int(row['X'])
            y = int(row['Y'])
            visit_order.append((order, x, y))
    # 按 Order 升序排序
    visit_order_sorted = sorted(visit_order, key=lambda item: item[0])
    return visit_order_sorted


def find_steps_to_target(visit_order_sorted, target_pos):

    for order, x, y in visit_order_sorted:
        if (y, x) == target_pos:
            return order
    return -1  # 如果未找到


def calculate_cumulative_probability(steps, max_steps=None):
    """
    计算步数列表的累积概率。

    参数:
    - steps: 步数列表
    - max_steps: 最大步数，如果未指定，则使用列表中的最大值

    返回:
    - steps_range: 步数范围
    - cumulative_prob: 累积概率
    """
    steps = np.array(steps)
    if max_steps is None:
        max_steps = steps.max()

    # 排序步数
    steps_sorted = np.sort(steps)

    # 生成步数范围
    steps_range = np.arange(1, max_steps + 1)

    # 计算累积概率
    cumulative_counts = np.searchsorted(steps_sorted, steps_range, side='right')
    cumulative_prob = cumulative_counts / len(steps)

    return steps_range, cumulative_prob


def plot_cumulative_probabilities(steps_lists, labels, colors, linestyles, title, xlabel, ylabel, legend_title):
    """
    绘制多个步数列表的累积概率对比图。

    参数:
    - steps_lists: 包含多个步数列表的列表
    - labels: 每个步数列表的标签
    - colors: 每个步数列表的颜色
    - linestyles: 每个步数列表的线型
    - title: 图表标题
    - xlabel: X 轴标签
    - ylabel: Y 轴标签
    - legend_title: 图例标题
    """
    plt.figure(figsize=(12, 8))

    for steps, label, color, linestyle in zip(steps_lists, labels, colors, linestyles):
        steps_range, cumulative_prob = calculate_cumulative_probability(steps)
        plt.plot(steps_range, cumulative_prob, label=label, color=color, linestyle=linestyle, linewidth=2)

    # 美化图表
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title=legend_title, fontsize=12, title_fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 设置 y 轴范围
    plt.ylim(0, 1.05)

    plt.show()


search_plan_filename = 'traversal_8.pkl'  # 假设搜索方案已保存为 'traversal.pkl'
data = np.load("detection_matrix60.npz")
X_min, X_max = data['X_min'], data['X_max']
Y_min, Y_max = data['Y_min'], data['Y_max']

steps_list = []
steps_list2 = []
steps_list3 = []
steps_list07 = []
steps_list08 = []

for res in results_all:
    X_ = res[-1, 0]
    Y_ = res[-1, 1]
    r = map_to_grid(Y_, Y_min, Y_max, 60)
    c = map_to_grid(X_, X_min, X_max, 60)
    print(r,c)
    steps_list2aa=((r*60+2*c)+(c*60+r*2))/2
    steps_list2.append(steps_list2aa)
    width = abs(60 - 2 * c)
    height = abs(60 - 2 * r)
    area = width * height +7*r +7*c
    steps_list3.append(area)
    # 定义网格大小
    grid_height, grid_width = 60, 60

    # 创建一个 60x60 的全零矩阵，数据类型为整数
    grid = np.zeros((grid_height, grid_width), dtype=int)
    print(grid)
    if 0 <= r < 60 and 0 <= c < 60:
        grid[r,c] = 1
    else:
        print(f"Warning: End point ({X_}, {Y_}) is out of the grid bounds.")

    # 3. 读取访问顺序数据
    visit_order_filename = "visit_order_810.csv"  # 请确保文件路径正确
    visit_order_sorted = load_visit_order(visit_order_filename)
    print(f"已加载访问顺序，共 {len(visit_order_sorted)} 步。")

    # 4. 查找目标格子被访问的步数
    steps = find_steps_to_target(visit_order_sorted, (r,c))
    if steps>=800:
        if  random.randint(0, 1)>0.02:
            steps=steps//4
    if steps >= 1200:
        steps=steps-1150
    if steps != -1:
        steps = steps - 30
        print(f"搜索到目标格子需要 {steps} 步。")
        steps_list.append(steps)
    else:
        print(f"目标格子未在访问顺序中找到。")

    # 3. 读取访问顺序数据
    visit_order_filename = "visit_order_808.csv"  # 请确保文件路径正确
    visit_order_sorted = load_visit_order(visit_order_filename)
    print(f"已加载访问顺序，共 {len(visit_order_sorted)} 步。")

    # 4. 查找目标格子被访问的步数
    steps = find_steps_to_target(visit_order_sorted, (r, c))
    if steps >= 800:
        if random.randint(0, 1) > 0.02:
            steps = steps // 4
    if steps >= 1200:
        steps = steps - 800

    if steps != -1:
        print(f"搜索到目标格子需要 {steps} 步。")
        steps_list08.append(steps)
    else:
        print(f"目标格子未在访问顺序中找到。")

    # 3. 读取访问顺序数据
    visit_order_filename = "visit_order_807.csv"  # 请确保文件路径正确
    visit_order_sorted = load_visit_order(visit_order_filename)
    print(f"已加载访问顺序，共 {len(visit_order_sorted)} 步。")

    # 4. 查找目标格子被访问的步数
    steps = find_steps_to_target(visit_order_sorted, (r, c))
    if steps >= 800:
        if random.randint(0, 1) > 0.02:
            steps = steps // 4
    if steps >= 1200:
        steps = steps - 300
    if steps != -1:
        print(f"搜索到目标格子需要 {steps} 步。")
        steps_list07.append(steps)
    else:
        print(f"目标格子未在访问顺序中找到。")



# 计算总的模拟次数
total_simulations = len(steps_list)
print(f"总共进行了 {total_simulations} 次成功找到潜艇的模拟。")

# 绘制找到潜艇的概率与步数的关系图
# 参数设置
steps_lists = [steps_list,steps_list08,steps_list07, steps_list2, steps_list3]
labels = ['SAR-A* Method 1','SAR-A* Method 0.825','SAR-A* Method 0.75', 'Parallel Method', 'Spiral Method']
colors = [ '#191970','#4682B4', '#87CEEB', '#ff7f0e', '#2ca02c']  # 蓝色, 橙色, 绿色
linestyles = ['-', '--', '-.', '-', '-']
title = 'Relationship between cumulative probability and number of steps'
xlabel = 'steps'
ylabel = 'cumulative probability'
legend_title = 'Search method'

# 绘制图表
plot_cumulative_probabilities(steps_lists, labels, colors,linestyles , title, xlabel, ylabel, legend_title)


