import torch
import argparse  # 导入argparse模块



import config.locomotion
from DDPG import DDPG, ReplayBuffer
from compare_search import action_search_bit
#from comper_ga import genetic_algorithm  # 

from formulation_tccn import *
import diffuser.utils as utils
import diffuser.sampling as sampling
from diffuser.datasets import Batch, ValueBatch
from drawing import single_plot_config

# 用来装载参数的容器
parser = argparse.ArgumentParser(description='Calculate volume of a cylinder')
# 给这个解析对象添加命令行参数
parser.add_argument('--experiment', type=str, help='Experiment name, diffusion, drl, search, ga or random',
                    default="diffusion")
parser.add_argument('--rounds', type=int, help=f'Running rounds, default {training_rounds}',
                    default=training_rounds)
parser.add_argument('--slots', type=int, help=f'Time slots per round, default {env_max_steps}',
                    default=env_max_steps)
parser.add_argument('--sp', type=bool, help=f'Show rounds pictures, default False', default=False)

rargs = parser.parse_args()
single_plot = single_plot_config(rargs.experiment)  # 直接传入实验名称如 "drl"



class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    # dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.locomotion'
    exp: str = ''


def main():
    q_env = TrajectoryENV(max_steps=20)



    if rargs.experiment == 'diffusion':
        args = Parser().parse_args('diffusion')
        observation_dim = q_env.state_dim
        action_dim = q_env.action_dim
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            horizon=args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=args.dim_mults,
            attention=args.attention,
            device=args.device,
        )

        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=args.n_diffusion_steps,
            loss_type=args.loss_type,
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            ## loss weighting
            action_weight=args.action_weight,
            loss_weights=args.loss_weights,
            loss_discount=args.loss_discount,
            device=args.device,
        )

        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config.pkl'),
            train_batch_size=args.batch_size,
            train_lr=args.learning_rate,
            gradient_accumulate_every=args.gradient_accumulate_every,
            ema_decay=args.ema_decay,
            sample_freq=args.sample_freq,
            save_freq=args.save_freq,
            label_freq=int(args.n_train_steps // args.n_saves),
            save_parallel=args.save_parallel,
            results_folder=args.savepath,
            bucket=args.bucket,
            n_reference=args.n_reference,
        )

        val_args = Parser().parse_args('values')
        val_model_config = utils.Config(
            val_args.model,
            savepath=(val_args.savepath, 'model_config.pkl'),
            horizon=val_args.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=val_args.dim_mults,
            device=val_args.device,
        )

        val_diffusion_config = utils.Config(
            val_args.diffusion,
            savepath=(val_args.savepath, 'diffusion_config.pkl'),
            horizon=val_args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=val_args.n_diffusion_steps,
            loss_type=val_args.loss_type,
            device=val_args.device,
        )

        val_model = val_model_config()
        val_diffusion = val_diffusion_config(val_model)

        model = model_config()
        diffusion = diffusion_config(model)

        trainer = trainer_config(diffusion)
        val_trainer = trainer_config(val_diffusion)

        guide_args = Parser().parse_args('plan')
        guide_config = utils.Config(guide_args.guide, model=val_diffusion, verbose=False)
        guide = guide_config()

        policy_config = utils.Config(
            guide_args.policy,
            guide=guide,
            scale=guide_args.scale,
            diffusion_model=diffusion,
            # normalizer=diffuser.datasets.normalization.GaussianNormalizer,
            normalizer=guide_args.normalizer,
            preprocess_fns=guide_args.preprocess_fns,
            ## sampling kwargs
            sample_fn=sampling.n_step_guided_p_sample,
            n_guide_steps=guide_args.n_guide_steps,
            t_stopgrad=guide_args.t_stopgrad,
            scale_grad_by_std=guide_args.scale_grad_by_std,
            verbose=False,
        )

        policy = policy_config()
        loss1_list, loss2_list = [], []
    elif rargs.experiment == 'drl':
        agent = DDPG(q_env.state_dim, q_env.action_dim, 1)
        loss1_list, loss2_list = [], []
    else:
        rargs.rounds = 1



    for e in range(rargs.rounds):

        observation = q_env.reset()
        q_list, e_list, r_list, a_list, rewards = [], [], [], [], []
        if rargs.experiment == 'drl':
            rb = ReplayBuffer(q_env.state_dim, q_env.action_dim)
        elif rargs.experiment == 'diffusion':
            conditions = []
            trajectories = []

        for q_round in range(rargs.slots):
            if rargs.experiment == 'diffusion':
                conditions.append(observation)
                action, trajectory = policy({0: observation}, batch_size=guide_args.batch_size,
                                            verbose=guide_args.verbose)
                next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = q_env.step(action)
            elif rargs.experiment == 'drl':
                action = agent.choose_action(observation)
                next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = q_env.step(action)
                rb.store(observation, action, reward, next_observation, terminal)
            elif rargs.experiment == 'search':
                action = action_search_bit(q_round, q_env)
                next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = q_env.step_int_act(action)
            elif rargs.experiment == 'random':
                action = np.random.randint(0, 2, q_env.action_dim) * 2
                next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = q_env.step_int_act(action)
            # elif rargs.experiment == 'ga':
            #     # 启动遗传算法实验
            #     action = genetic_algorithm(q_env, generations=100, pop_size=20, mutation_rate=0.1)
            #     next_observation, reward, terminal, et, q_sum, ru_avg, a_sum = q_env.step_int_act(action)


            # print("reward", reward)
            if rargs.experiment == 'diffusion':
                trajectories.append(trajectory[0])
                rewards.append([reward])
            else:
                rewards.append(reward)

            observation = next_observation
            q_list.append(q_sum)
            e_list.append(et)
            r_list.append(ru_avg)
            a_list.append(a_sum)



       
        x_label = "time slot"
        single_plot(q_list, x_label, "qu avg", "qu_avg-arriving_step", str(e), show_pic=rargs.sp)
        single_plot(e_list, x_label, "eu avg", "eu_avg-arriving_step", str(e), show_pic=rargs.sp)
        single_plot(r_list, x_label, "ru avg", "ru_avg-arriving_step", str(e), show_pic=rargs.sp)
        single_plot(a_list, x_label, "Au avg", "au_avg-arriving_step", str(e), show_pic=rargs.sp)
        single_plot(np.reshape(np.array(rewards), -1), x_label, "reward", "reward-arriving_step", str(e),
                    show_pic=rargs.sp)
        if rargs.experiment == "diffusion":
            def tt(arr):
                return torch.FloatTensor(np.array(arr, dtype=np.float32))

            conditions = {0: tt(conditions)}
            trajectories = tt(trajectories)
            values = tt(rewards)
            loss1_list.append(trainer.my_train(Batch(conditions=conditions, trajectories=trajectories), 'diffusion:'))
            loss2_list.append(val_trainer.my_train(
                ValueBatch(conditions=conditions, trajectories=trajectories, values=values), "val diffusion:"))
        elif rargs.experiment == "drl":
            aloss, bloss = agent.learn(rb)
            print(f"round:{e}, actor loss:{aloss}, critic loss:{bloss}")
            loss1_list.append(aloss)
            loss2_list.append(bloss)

    
    if rargs.experiment in ['diffusion', 'drl']:
        q_env.save_trajectory(f"./logs/{rargs.experiment}_trajectory.npy")

   
    if rargs.experiment == "diffusion" or rargs.experiment == "drl":
        x_label = "training rounds"
        single_plot(loss1_list, x_label, "diffusion loss", "diffusion_loss-rounds", "last", show_pic=True)
        single_plot(loss2_list, x_label, "val diffusion loss", "val_diffusion_loss-rounds", "last", show_pic=True)


if __name__ == '__main__':
    main()

