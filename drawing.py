import os

import matplotlib.pyplot as plt
import numpy as np

from formulations import *

new_time_factor = 0.001
font_dict = {"size": 12}

figure_type = "png"


def get_e_config(u=users, f=f_u_max_ori):
    return f"users_{u}-f_u_max-{f}"


def single_plot_config(path):
    op = os.path.join("./logs", path, get_e_config())
    if not os.path.exists(op):
        os.makedirs(op)

    def single_plot(y_axis, x_label, y_label, f_name, pr, title=None, show_pic=False):
        od = os.path.join(op, pr)
        if not os.path.exists(od):
            os.mkdir(od)
        np.savetxt(os.path.join(od, f"{f_name}.txt"), y_axis)
        if show_pic:
            # with plt.style.context(['science', 'ieee']):
            plt.xlabel(x_label, font_dict)
            plt.ylabel(y_label, font_dict)
            plt.plot(np.arange(1, len(y_axis) + 1), y_axis, linestyle="-")
            plt.title(f"round:{pr}")
            plt.savefig(os.path.join(od, f"{f_name}.{figure_type}"))
            plt.show()

    return single_plot


def load_experiments_data(experiment, u=users, f=f_u_max_ori):
    path = os.path.join("./logs", experiments_config[experiment]["file_folder"], get_e_config(u, f))
    exp_times = len(os.listdir(path))
    qu_list, eu_list, ru_list, au_list, rewards = [], [], [], [], []

    for i in range(exp_times):
        exp_path = os.path.join(path, str(i))
        qu_list.append(np.loadtxt(os.path.join(exp_path, "qu_avg-arriving_step.txt")))
        eu_list.append(np.loadtxt(os.path.join(exp_path, "eu_avg-arriving_step.txt")))
        ru_list.append(np.loadtxt(os.path.join(exp_path, "ru_avg-arriving_step.txt")))
        au_list.append(np.loadtxt(os.path.join(exp_path, "au_avg-arriving_step.txt")))
        rewards.append(np.loadtxt(os.path.join(exp_path, "reward-arriving_step.txt")))
    return np.array(qu_list), np.array(eu_list), np.array(ru_list), np.array(au_list), np.array(rewards)


if __name__ == "__main__":
    arr, _, _, _, _ = load_experiments_data("diffusion")
    print(arr, np.shape(arr))
