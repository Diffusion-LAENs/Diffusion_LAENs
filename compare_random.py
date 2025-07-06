from formulations import *
from drawing import single_plot_config


def main():
    single_plot = single_plot_config(experiments_config["random"]["file_folder"])

    q_env = ENV(max_steps=env_max_steps)

    loss1_list, loss2_list = [], []
    for e in range(training_rounds):
        q_list, e_list, r_list, a_list, rewards = [], [], [], [], []
        for q_round in range(env_max_steps):
            action = np.random.randint(0, 2, users) * 2
            next_observation, reward, terminal, et, q_sum, psi_sum, a_sum = q_env.step_int_act(action)
            # print("round:", e, ",reward:", reward)

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
    # print(np.random.randint(0,2,users)*2-1)
