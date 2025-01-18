import numpy as np
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pickle  # 导入 pickle 模块用于序列化

class SAR_A_Star_Square:
    def __init__(self, grid_height, grid_width, p_DT_map, start=(0, 0), neighbor_mode='4'):
        """
        初始化 SAR-A* 算法实例。
        - grid_height, grid_width: 网格的高度和宽度
        - p_DT_map: 目标发现概率矩阵 (二维 NumPy 数组)
        - start: 起始位置 (row, col)
        - neighbor_mode: 邻接模式，'4' 表示上下左右，'8' 表示包含对角线
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.p_DT_map = p_DT_map
        self.start_position = start
        self.neighbor_mode = neighbor_mode

        # 初始化路径和访问顺序
        self.path = []
        self.visit_order = {}
        self.current_order = 1  # 访问顺序计数器

    def euclidean_dist(self, pos1, pos2):
        """
        计算 pos1 与 pos2 间的欧几里得距离。
        pos1, pos2: 均为 (row, col) 格式的元组
        """
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def angle_between(self, p_prev, p_curr, p_next):
        """
        计算 (p_prev -> p_curr) 与 (p_curr -> p_next) 两个向量的夹角(弧度)。
        若 p_prev 为空或与 p_curr 相同，则认为转角为 0。
        """
        if p_prev is None or p_prev == p_curr:
            return 0.0

        # 向量 (p_prev -> p_curr) 和 (p_curr -> p_next)
        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        # 计算夹角
        dot = v1[0] * v2[0] + v1[1] * v2[1]  # 点乘
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return 0.0
        cos_ = dot / (mag1 * mag2)
        # 数值稳定处理
        cos_ = max(min(cos_, 1.0), -1.0)
        return math.acos(cos_)

    def get_neighbors(self, pos):
        """
        获取正方形网格中某单元的邻居列表。
        mode='4' 表示上下左右四邻接；mode='8' 表示包括对角线的八邻接。
        """
        (r, c) = pos
        candidates = []

        if self.neighbor_mode == '4':
            # 上下左右
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                    candidates.append((nr, nc))
        elif self.neighbor_mode == '8':
            # 8 邻接
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                        candidates.append((nr, nc))
        else:
            raise ValueError("neighbor_mode must be '4' or '8'")
        return candidates

    def f1_distance_cost(self, curr_pos, next_pos, best_pos):
        """
        对应文中 f1: 考虑与当前单元 (curr_pos)
        以及目标发现概率最高单元 (best_pos) 的距离。
        """
        # 单位距离 d_m 根据邻接模式调整
        if self.neighbor_mode == '4':
            d_m = 1.0
        else:  # '8'
            d_m = math.sqrt(2)

        w_d = 0.5
        w_db = 0.5
        dist_curr = self.euclidean_dist(curr_pos, next_pos) / d_m
        dist_best = self.euclidean_dist(best_pos, next_pos) / d_m
        return w_d * dist_curr + w_db * dist_best

    def f2_turning_cost(self, p_parent, p_curr, p_next):
        """
        对应文中 f2: 转向角越大，代价值越高。
        这里返回转角作为代价(越小越好)，可以根据需要二次处理。
        """
        angle_val = self.angle_between(p_parent, p_curr, p_next)
        # 也可对 angle_val 做归一化，例如 / pi 等
        return angle_val  # 越小越好

    def f3_probability_gain(self, next_pos):
        """
        对应文中 f3: p_DT 越大越好。
        这里为了方便，可以返回一个 "负代价" 或者用 1 - 归一化(p_DT)。
        """
        # 原文是 max f3 => 这里可以返回 1 - p_DT 以适配 "min cost" 写法
        return 1.0 - self.p_DT_map[next_pos[0], next_pos[1]]

    def combined_heuristic(self, p_parent, p_curr, next_pos, best_pos,
                           w1=1.0, w2=1.0, w3=1.0):
        """
        将 f1, f2, f3 三目标加权求和，这里用简单加权和做示例。
        - w1, w2, w3 是对应 f1, f2, f3 的权重。
        """
        cost_f1 = self.f1_distance_cost(p_curr, next_pos, best_pos)
        cost_f2 = self.f2_turning_cost(p_parent, p_curr, next_pos)
        cost_f3 = self.f3_probability_gain(next_pos)  # 注意 f3 这里是 "越小越好" 需要转换

        # 简单做一个线性加权和
        cost = w1 * cost_f1 + w2 * cost_f2 + w3 * cost_f3
        return cost

    def find_best_pDT(self, open_list):
        """
        从 open_list 中找出具有最大 p_DT 的网格位置。
        """
        best_cell = None
        best_val = -1.0
        for cell in open_list:
            val = self.p_DT_map[cell[0], cell[1]]
            if val > best_val:
                best_val = val
                best_cell = cell
        return best_cell

    def bfs_path_search(self, start, goal):
        """
        在一个无障碍的 grid 上，使用 BFS 寻找从 start 到 goal 的最短路径。
        - start, goal: (row, col)
        返回：若可达则返回路径(含首尾)；不可达返回 None
        """
        queue = deque()
        queue.append(start)
        visited_bfs = set([start])
        parent = dict()  # 用于重建路径

        while queue:
            current = queue.popleft()
            if current == goal:
                # 找到目标，回溯路径
                path = []
                node = goal
                while node is not None:
                    path.append(node)
                    node = parent.get(node, None)
                path.reverse()
                return path  # 返回从 start -> goal 的路径

            # 获取邻居
            for n in self.get_neighbors(current):
                if n not in visited_bfs:
                    visited_bfs.add(n)
                    parent[n] = current
                    queue.append(n)

        # 若 BFS 结束仍未找到 goal，则返回 None
        return None

    def sar_a_star_square(self):
        """
        基于正方形网格的 SAR-A* 主函数 (含 BFS passing 的版本)。
        """
        # 所有格子构成 openList
        open_list = set()
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                open_list.add((r, c))

        # 已访问列表
        visited = set()

        # 最终生成的访问路径
        self.path = []

        # 记录访问顺序
        self.visit_order = {}
        self.current_order = 1

        # 当前单元设定为 start
        current = self.start_position
        parent_cell = None  # 用于计算转向角

        # 确保 start 在网格范围内
        if current not in open_list:
            raise ValueError("start 不在网格范围内！")

        while open_list:
            # 将当前单元加入路径 & 已访问
            self.path.append(current)
            self.visit_order[current] = self.current_order
            self.current_order += 1
            visited.add(current)
            open_list.discard(current)

            # 计算 p_DT 最大的单元位置
            best_cell = self.find_best_pDT(open_list) if open_list else current

            # 获取可用邻居
            neighbors = self.get_neighbors(current)
            available_list = [n for n in neighbors if n in open_list]

            if len(available_list) > 0:
                # 存在可用邻居 => 启发式函数 h 选最优
                costs = []
                for cand in available_list:
                    cost_val = self.combined_heuristic(
                        parent_cell, current, cand, best_cell,
                        w1=1.0, w2=1.0, w3=1.0
                    )
                    costs.append((cand, cost_val))
                # 选 cost 最小的那个
                costs.sort(key=lambda x: x[1])
                next_cell = costs[0][0]

                # 更新 parent_cell，转移到 next_cell
                parent_cell = current
                current = next_cell

            else:
                # 没有可用邻居，但仍然有未访问单元
                if len(open_list) > 0:
                    next_target = self.find_best_pDT(open_list)

                    # 使用 BFS 查找从 current -> next_target 的最短路径
                    bfs_result = self.bfs_path_search(current, next_target)
                    if bfs_result is None:
                        # 若没有路径(理论上无障碍则不会发生)
                        print("Warning: BFS 未找到通路，强行跳转")
                        self.path.append(next_target)
                        self.visit_order[next_target] = self.current_order
                        self.current_order += 1
                        open_list.discard(next_target)
                        parent_cell = None
                        current = next_target
                    else:
                        # 将 BFS 路径上的每个点都加入 path
                        # 跳过第一个点 (current)
                        for idx, pnt in enumerate(bfs_result):
                            if idx == 0:
                                continue  # 跳过当前起点以避免重复
                            self.path.append(pnt)
                            self.visit_order[pnt] = self.current_order
                            self.current_order += 1
                            visited.add(pnt)
                            if pnt in open_list:
                                open_list.remove(pnt)
                        # 最终到达 next_target
                        if len(bfs_result) > 1:
                            parent_cell = bfs_result[-2]
                        else:
                            parent_cell = None
                        current = bfs_result[-1]

                else:
                    # 完全覆盖结束
                    break

    def visualize_path(self):
        """
        可视化 SAR 路径和检测概率。
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.p_DT_map, cmap='viridis', origin='upper')
        plt.colorbar(label='Detection Probability')

        # 绘制路径
        if len(self.path) > 0:
            path_rows, path_cols = zip(*self.path)
            plt.plot(path_cols, path_rows, marker='o', color='red', label='SAR Path', linewidth=1)

            # 标记起点与终点
            plt.scatter(self.start_position[1], self.start_position[0], color='blue', label='Start Position', s=100)
            plt.scatter(self.path[-1][1], self.path[-1][0], color='green', label='End Position', s=100)

        plt.title('SAR Path Visualization')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        # 添加图例并固定在右下角
        plt.legend(fontsize=12, loc='lower right')
        plt.show()

    def visualize_path_double(self):
        """
        可视化 SAR 路径和检测概率，将其分为左右两个子图：
        - 左边绘制检测概率图
        - 右边绘制路线图
        """
        # 创建一个包含两个子图的图形，左右排列
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # -------------------
        # 左边子图：检测概率图
        # -------------------
        im1 = axs[0].imshow(self.p_DT_map, cmap='viridis', origin='upper')
        axs[0].set_title('Detection Probability Map', fontsize=16)
        axs[0].set_xlabel('Column Index', fontsize=14)
        axs[0].set_ylabel('Row Index', fontsize=14)

        # 添加颜色条
        cbar = fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
        cbar.set_label('Detection Probability', fontsize=14)

        # -------------------
        # 右边子图：路线图
        # -------------------
        # 设置右侧子图的背景颜色为白色
        axs[1].set_facecolor('white')

        if len(self.path) > 0:
            # 解压路径的行和列
            path_rows, path_cols = zip(*self.path)

            total_len = len(self.path)
            quarter = total_len // 4  # 每段的长度
            colors = ['skyblue', 'yellowgreen', 'yellow', 'red']  # 定义四种颜色

            # 绘制四个颜色段的路径
            for i in range(4):

                start_idx = i * quarter
                # 确保最后一段包括所有剩余的点
                if i == 3:
                    end_idx = total_len
                else:
                    end_idx = (i + 1) * quarter +1

                # 确保每段至少有一个点
                if start_idx >= end_idx:
                    continue

                # 为第一个段落添加标签，以便图例中只显示一次
                if i == 0:
                    axs[1].plot(
                        path_cols[start_idx:end_idx],
                        path_rows[start_idx:end_idx],
                        marker='',
                        color=colors[i],
                        label='SAR Path',
                        linewidth=2
                    )
                else:
                    axs[1].plot(
                        path_cols[start_idx:end_idx],
                        path_rows[start_idx:end_idx],
                        marker='',
                        color=colors[i],
                        linewidth=2
                    )

            # 标记起点与终点
            axs[1].scatter(
                self.start_position[1],
                self.start_position[0],
                color='blue',
                label='Start Position',
                s=100
            )
            axs[1].scatter(
                self.path[-1][1],
                self.path[-1][0],
                color='green',
                label='End Position',
                s=100
            )

        axs[1].set_xlabel('Column Index', fontsize=14)
        axs[1].set_ylabel('Row Index', fontsize=14)

        # 添加图例并固定在右下角
        axs[1].legend(fontsize=12, loc='lower right')

        # 调整布局以防止子图重叠
        plt.tight_layout()

        # 显示图形
        plt.show()

    def visualize_visit_order(self):
        """
        可视化网格单元的访问顺序。
        """
        # 创建一个矩阵来存储访问顺序
        visit_matrix = np.zeros((self.grid_height, self.grid_width))

        # 填充访问顺序
        for (r, c), order in self.visit_order.items():
            visit_matrix[r, c] = order

        # 绘制访问顺序矩阵
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.coolwarm
        cmap.set_under(color='white')
        norm = Normalize(vmin=1, vmax=self.current_order)
        plt.imshow(visit_matrix, cmap=cmap, origin='upper', norm=norm)
        plt.colorbar(label='Visit Order')
        plt.title('Grid Cell Visit Order')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.show()

    def run_demo(self):
        """
        运行一个示例场景并进行可视化。
        """
        # 执行 SAR-A* 算法
        self.sar_a_star_square()

        # 输出结果
        print("访问顺序 path 长度:", len(self.path))
        print("访问顺序 path:", self.path)

        # (可选) 查看最终是否覆盖全部网格
        covered_count = len(set(self.path))
        total_cells = self.grid_height * self.grid_width
        print(f"覆盖格子数 = {covered_count} / {total_cells} (已访问比例 = {covered_count / total_cells:.2f})")

        # 可视化路径
        self.visualize_path()
        self.visualize_path_double()

        # 可视化访问顺序
        self.visualize_visit_order()

def load_detection_matrix(filename="detection_matrix60.npz"):
    data = np.load(filename)
    print(f"Detection matrix loaded from {filename}")
    detection_matrix = data['detection_matrix']

    return detection_matrix

# 示例运行
def demo_sar_astar():
    """
    演示入口：构造一个示例场景并运行 SAR-A*。
    """
    # 1) 网格大小
    H, W = 60, 60

    # 2) 构造目标发现概率 p_DT_map
    p_DT_map = load_detection_matrix()  # 0~1 随机


    # 3) 设定起点 (row=0, col=0)
    start = (0, 0)

    # 4) 创建 SAR-A* 实例并运行（4邻接）
    print("=== 使用 4 邻接模式 ===")
    sar_4 = SAR_A_Star_Square(grid_height=H, grid_width=W, p_DT_map=p_DT_map, start=start, neighbor_mode='4')
    sar_4.run_demo()

    # 5) 创建 SAR-A* 实例并运行（8邻接）
    print("\n=== 使用 8 邻接模式 ===")
    sar_8 = SAR_A_Star_Square(grid_height=H, grid_width=W, p_DT_map=p_DT_map, start=start, neighbor_mode='8')
    sar_8.run_demo()


if __name__ == "__main__":
    demo_sar_astar()
