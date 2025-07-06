import numpy as np
from scipy.optimize import minimize

env_max_steps = 20 #十个时间步数
training_rounds = 300 #训练伦次
experiments_config = {"diffusion": {'title': 'Diffusion', 'file_folder': 'diffusion'},
                      "drl": {'title': 'DRL', 'file_folder': 'compare_drl'},
                      "search": {'title': 'Search', 'file_folder': 'compare_search'},
                      "random": {'title': 'Random', 'file_folder': 'compare_random'},
"ga": {'title':       "GA", 'file_folder': 'compare_ga'}  # 新增遗传算法实验配置
                      }

uavs = 1#待确定  无人机的数量  图8的横坐标变量
evtols = 1#待确定

# 参数
Bm = 7e7
Bn = 7e7
sigma2 = 1e-9
Pm_max = 1.5    # 最大发射功率 eVTOL 0.3
Pn_max = 1.5     # 最大发射功率 UAV  0.3
hm = 900     # cpu cycles per bit of data 待确定  evtol的本地计算能力 图9的横坐标
hn = 900  # cpu cycles per bit of data 待确定  UAV的本地计算能力 图9的横坐标
kappa_m = 1e-28
kappa_n = 1e-28
kappa1 = 0.0661
kappa2 = 15.97
omega_b = 0.3
#m = 1500000 #evtol = 1500kg
zeta = 9.8
L_D = 12
eta = 0.85
eta_m = 1.2
eta_n = 1.0
V = 4e12
vmax_m = 30
vmax_n = 30
rate_thresh = 1e5  # 可调

#f_u_max = [0.5, 1, 1.5, 2, 2.5]  # GHz
f_u_max_ori = 1.5  # GHz
f_u_max = f_u_max_ori * 10 ** 9  # Hz



class TrajectoryENV:
    def __init__(self, max_steps=100):
        self.M = evtols
        self.N = uavs
        self.max_steps = max_steps
        self.tau = 1

        self.dim = 2  # 2D 平面飞行
        self.total = self.M + self.N
        self.state_dim = self.total * (2 + 2 )  # p, v, θ,  位置、速度、角度
        self.action_dim = self.total * 2  # 控制 [v̇, θ̇]  加速度与角速度

        self.reset()

    def reset(self):
        self.step_num = 0

        # eVTOL 初始位置 (-5000, 随机y)
        self.pm = np.array([[-5000, np.random.uniform(-1000, 1000)] for _ in range(self.M)], dtype=np.float64)

        # UAV 初始位置 (-5000, 随机y)
        self.pn = np.array([[-5000, np.random.uniform(-1000, 1000)] for _ in range(self.N)], dtype=np.float64)

        # 记录终点位置 (+5000, 随机y)
        self.goal_pm = np.array([[5000, np.random.uniform(-1000, 1000)] for _ in range(self.M)], dtype=np.float64)
        self.goal_pn = np.array([[5000, np.random.uniform(-1000, 1000)] for _ in range(self.N)], dtype=np.float64)

        self.trajectory = {"evtol": [], "uav": []}  # 新增

        self.vm = np.full((self.M,), 60, dtype=np.float64)  # 固定初始速度 60 m/s
        self.vn = np.full((self.N,), 1, dtype=np.float64)  # 固定初始速度 1 m/s
        self.thetam = np.zeros((self.M,), dtype=np.float64)
        self.thetan = np.zeros((self.N,), dtype=np.float64)

        self.qm = np.random.uniform(5e6, 10e6, self.M)
        self.qn = np.random.uniform(5e6, 10e6, self.N)

        return self.get_state()

    def get_state(self):
        state = np.concatenate([
            self.pm.flatten(), self.pn.flatten(),
            self.vm, self.vn,
            self.thetam, self.thetan,
            #self.qm, self.qn,
        ])
        return state

    def get_channel_gain(self, p):
        H = 100  # 高度
        gu = 1e-5
        pb = np.array([0.0, 0.0])
        dist2 = np.sum((p - pb) ** 2, axis=1) + H ** 2
        return gu / dist2


    #优化功率函数
    def optimize_power(self, q, g, B, P_max):
        # 通信功率优化：求解 argmin_P V*E - Q * R
        def objective(P):
            R = B * np.log2(1 + P * g / sigma2)
            return V * self.tau * P - q * (self.tau * R)

        bounds = [(0, P_max)] * len(q)
        res = minimize(lambda P: np.sum(objective(P)), x0=np.ones_like(q) * 0.1, bounds=bounds)
        return res.x



    def compute_energy(self, vm, vn, f_m, f_n, Pm, Pn, xm, xn):
        Emf = self.tau  / (eta * omega_b) * ( zeta * vm / L_D) * 200 #50kg为电池的重量
        Enf = self.tau * (kappa1 * vn ** 3 + kappa2 / vn)
        Eml = self.tau * kappa_m * f_m ** 3
        Enl = self.tau * kappa_n * f_n ** 3
        Emb = self.tau * Pm
        Enb = self.tau * Pn
        Em = Emf + (1 - xm) * Eml + xm * Emb
        En = Enf + (1 - xn) * Enl + xn * Enb


        E = eta_m * np.sum(Em) + eta_n * np.sum(En)
        #print(f"[debug] vm = {vm}, vn = {vn}, Emf = {Emf}, Enf = {Enf}, Eml = {Eml}, Enl = {Enl}, Emb = {Emb},  Enb = { Enb}, total E = {E}")

        # 🔧 修复：避免 shape 不一致相加，使用 sum
        return E

    def step(self, action):
        # action: [Δv₁, Δθ₁, ..., Δv_M+N, Δθ_M+N]
        dv = action[::2]  # shape = (M + N,)
        dtheta = action[1::2]  # shape = (M + N,)

        dv_m = dv[:self.M]
        dv_n = dv[self.M:]
        dtheta_m = dtheta[:self.M]
        dtheta_n = dtheta[self.M:]

        # 分别更新 eVTOL 和 UAV 的速度和角度
        self.vm = np.clip(self.vm + dv_m, 20, vmax_m).astype(np.float64)  # 添加类型转换
        self.vn = np.clip(self.vn + dv_n, 20, vmax_n).astype(np.float64)  # 添加类型转换

        self.thetam = (self.thetam + dtheta_m).astype(np.float64)  # 添加类型转换
        self.thetan = (self.thetan + dtheta_n).astype(np.float64)  # 添加类型转换

        # 更新位置（显式类型转换）
        delta_pm = self.tau * self.vm[:, None] * np.stack([
            np.cos(self.thetam),
            np.sin(self.thetam)
        ], axis=1).astype(np.float64)

        delta_pn = self.tau * self.vn[:, None] * np.stack([
            np.cos(self.thetan),
            np.sin(self.thetan)
        ], axis=1).astype(np.float64)

        # 强制类型转换后再相加
        self.pm = self.pm.astype(np.float64) + delta_pm
        self.pn = self.pn.astype(np.float64) + delta_pn

        self.trajectory["evtol"].append(self.pm.copy())
        self.trajectory["uav"].append(self.pn.copy())


        # compute gain
        g_m = self.get_channel_gain(self.pm)
        g_n = self.get_channel_gain(self.pn)

        # ⭐ 动态发射功率优化
        Pm = self.optimize_power(self.qm, g_m, Bm, Pm_max)
        Pn = self.optimize_power(self.qn, g_n, Bn, Pn_max)

        #print(f"[debug] Pm = {Pm}, Pn = {Pn}")

        # simple rate model
        Rm = Bm * np.log2(1 + Pm * g_m / sigma2)
        Rn = Bn * np.log2(1 + Pn * g_n / sigma2)

        nm = self.tau * Rm
        nn = self.tau * Rn

        f_m = np.minimum(np.sqrt(self.qm / (3 * V * kappa_m * hm)),f_u_max)
        f_n = np.minimum(np.sqrt(self.qn / (3 * V * kappa_n * hn)),f_u_max)

        Dm = self.tau * f_m / hm
        Dn = self.tau * f_n / hn

        # Offloading decision: 卸载判决：基于优化后的速率是否高于门限
        xm = (Rm > rate_thresh).astype(int)
        xn = (Rn > rate_thresh).astype(int)

        # energy
        E = self.compute_energy(self.vm, self.vn, f_m, f_n, Pm, Pn, xm, xn)


        # queue update
        Ψm = (1 - xm) * Dm + xm * nm
        Ψn = (1 - xn) * Dn + xn * nn

        Am = np.random.uniform(5e5, 10e5, self.M)
        An = np.random.uniform(5e5, 10e5, self.N)

        self.qm = np.maximum(self.qm - Ψm, 0) + Am
        self.qn = np.maximum(self.qn - Ψn, 0) + An

        Q_all = np.concatenate([self.qm, self.qn])
        A_all = np.concatenate([Am, An])

        queue_term_m = np.sum(self.qm * (Ψm - Am))
        queue_term_n = np.sum(self.qn * (Ψn - An))


        raw_reward = V * np.sum(E) - queue_term_m - queue_term_n
        reward = raw_reward / 1e15  




        # Debug 打印
        #print(f"[debug] raw reward: {raw_reward:.2e}")
        #print(f"[debug] scaled reward: {scaled_reward:.2e}")

        # 计算到终点的距离
        dist_pm = np.linalg.norm(self.pm - self.goal_pm, axis=1)
        dist_pn = np.linalg.norm(self.pn - self.goal_pn, axis=1)

        # 判断是否到达终点
        done = self.step_num >= self.max_steps or np.all(dist_pm < 50) or np.all(dist_pn < 50)

        self.last_energy = np.mean(E)
        self.last_reward = reward

        return self.get_state(), reward, done, np.mean(E), np.mean(Q_all), np.mean(np.concatenate([Rm, Rn])), np.mean(
            A_all)

    def save_trajectory(self, filename="flight_trajectory.npy"):
        np.save(filename, self.trajectory)
        print(f"[Trajectory] Saved to {filename}")




    def step_int_act(self, int_action, update=True):
        """
        整数动作版本的step函数。用于搜索/随机策略中的离散动作表示。
        将整型动作值（如0,1,2）映射为连续控制变量（加速度、角速度）。
        """
        # 动作维度为 total * 2（每个UAV/eVTOL一个[Δv, Δθ]）
        # 允许的离散动作为：0 → -1, 1 → 0, 2 → 1
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        action = np.array([mapping.get(a, 0.0) for a in int_action])

        next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = self.step(action)
        return next_observation, reward, terminal, et, q_sum, ru_avg, a_sum


def print_step_info(self):
    print("Step:", self.step_num)
    print("Mean queue length:", np.mean(np.concatenate([self.qm, self.qn])))
    print("Mean energy:", self.last_energy)
    print("Reward:", self.last_reward)








if __name__ == '__main__':
    env = TrajectoryENV()
    s = env.reset()
    for i in range(5):
        a = np.random.randn(env.action_dim)
        s_, r, d, e, q, ru, au = env.step(a)
        print(f"step {i} → reward: {r:.2f}, energy: {e:.2f}, queue: {q:.2e}")
