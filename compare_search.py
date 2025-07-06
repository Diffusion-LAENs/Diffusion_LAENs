import sys
import multiprocessing as mp

import numpy as np

from formulations import *
from drawing import single_plot_config


def dec_to_ter(num, length):
    l = []
    if num < 0:
        return "- " + dec_to_ter(abs(num), length)  # 负数先转为正数，再调用函数主体
    else:
        while True:
            num, reminder = divmod(num, 3)  # 算除法求除数和余数
            l.append(reminder)  # 将余数存入字符串
            if num == 0:
                for i in range(length - len(l)):
                    l.append(0)
                return np.array(l[::-1])


def action_search(r, q_env):
    n = q_env.action_dim
    round_val = mp.Value('i', 0)
    min_x = mp.Array('f', np.zeros(n))
    min_val = mp.Value('f', sys.maxsize)

    bounds = np.zeros((n, 2))
    bounds[:, 1] = 1
    bounds[:, 0] = -1
    # print("bounds", bounds)
    # eq_cons = {'type': 'eq',
    #            'fun': lambda x: sum(x) - 1}

    process_num = 5

    def mp_func(idx):
        def rosen(x):
            next_observation, reward, terminal, et, q_sum, psi_sum, a_sum = q_env.step(x, update=False)
            round_val.value = round_val.value + 1
            if reward < min_val.value:
                min_val.value = reward
                min_x[:] = x
                print("round", r, f",search:{round_val.value}", ",R(O(t)):", reward, ",action", x)
            return reward

        edge_x0 = np.ones(n) * 2 * idx / process_num - 0.8
        edge_x0[:int(n / 2)] = -edge_x0[:int(n / 2)]
        # print("start", edge_x0)
        minimize(rosen, edge_x0, method='trust-constr',
                 options={'maxiter': 20, 'verbose': 0, 'disp': False},
                 bounds=bounds)

    processes = []
    for i in range(process_num):
        p = mp.Process(target=mp_func, args=(i,))
        p.start()
        processes.append(p)
    [p.join() for p in processes]
    # print("mp rounds", round_val.value)
    return np.array(min_x)


def action_search_bit(r, q_env):
    n = q_env.action_dim
    round_val = mp.Value('i', 0)
    min_x = mp.Array('f', np.zeros(n))
    min_val = mp.Value('f', sys.maxsize)
    solve_size = 3 ** n

    process_num = 5

    def mp_func(idx):
        start = int(idx * solve_size / process_num)
        for sn in range(10):
            action = dec_to_ter(start + sn, n)
            next_observation, reward, terminal, et, q_sum, psi_sum, a_sum = q_env.step_int_act(action, update=False)
            round_val.value = round_val.value + 1
            if reward < min_val.value:
                min_val.value = reward
                min_x[:] = action
                print("time slots:", r, f",search:{round_val.value}", ",R(O(t)):", reward, ",action", action)

    processes = []
    for i in range(process_num):
        p = mp.Process(target=mp_func, args=(i,))
        p.start()
        processes.append(p)
    [p.join() for p in processes]
    # print("mp rounds", round_val.value)
    return np.array(min_x)


def main():
    single_plot = single_plot_config(experiments_config["search"]["file_folder"])

    q_env = ENV(max_steps=env_max_steps)

    loss1_list, loss2_list = [], []
    for e in range(training_rounds):
        q_list, e_list, r_list, a_list, rewards = [], [], [], [], []
        for q_round in range(env_max_steps):
            action = action_search_bit(e, q_env)
            next_observation, reward, terminal, et, q_sum, psi_sum, a_sum = q_env.step_int_act(action)
            print("round:", e, ",reward:", reward)

            # results
            q_list.append(q_sum)
            e_list.append(et)
            r_list.append(psi_sum)
            a_list.append(a_sum)
            rewards.append(reward)

        x_label = "data arriving step"
        show1 = False
        single_plot(q_list, x_label, "qu avg", "qu_avg-arriving_step", str(e))
        single_plot(e_list, x_label, "eu avg", "eu_avg-arriving_step", str(e))
        single_plot(r_list, x_label, "ru avg", "ru_avg-arriving_step", str(e))
        single_plot(a_list, x_label, "Au avg", "au_avg-arriving_step", str(e))
        single_plot(rewards, x_label, "reward", "reward-arriving_step", str(e), show_pic=show1)

    x_label = "training rounds"
    single_plot(loss1_list, x_label, "actor loss", "diffusion_loss-rounds", "last", show_pic=True)
    single_plot(loss2_list, x_label, "critic loss", "val_diffusion_loss-rounds", "last", show_pic=True)


if __name__ == '__main__':
    main()
    # print(dec_to_ter(10,10))
