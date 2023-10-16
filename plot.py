import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import numpy as np
from pathlib import Path

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 12})
# rc('text', usetex=True)
ylabels={'online_reward_diff (r - rbaseline)': 'Reward(RL) - Reward(baseline)',
        'online_max voltage violation': 'Maximum voltage violation (volt)',
        'online_reward': 'Reward',
        'online_volt_reward': 'Voltage reward',
        'online_loss_reward': 'Loss reward',
        'online_action_reward': 'Action reward',
        'online_reward_baseline': 'Reward (baseline)',
        'online_action_reward_baseline': 'Action reward (baseline)',
        'online_volt_reward_baseline': 'Voltage reward (baseline)',
        'online_loss_reward_baseline': 'Loss reward (baseline)',}

def smooth(y, box_pts):
    # smooth curves by moving average
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def plot_res(envs, algos, smoothing, reward_option, state_option, xlabel):
    results = {}
    for ax, env in enumerate(envs):
        results[env] = {}
        for a in algos:
            path = Path('./res/data/state_{}/reward_{}/{}_{}.pkl'.format(state_option,
                                                                     reward_option,
                                                                     env,
                                                                     a))
            with open(path, 'rb') as f:
                results[env][a] = pickle.load(f)

    for metric in ylabels.keys():
        if "_baseline" in metric:
            continue
        fig, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.))
        for ax, env in enumerate(envs):

            v_all = []
            for res in results[env].values():
                if res[metric]:
                    v_all.append(np.array(res[metric]))
                else:
                    v_all.append([])

            for i, v_each in enumerate(v_all):
                if isinstance(v_each, np.ndarray):
                    axes[ax].plot(np.percentile(v_each, q=50, axis=0), label=algos[i])
                    axes[ax].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                        y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                        y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)
            if metric+"_baseline" in res.keys():
                v_each = np.array(res[metric+"_baseline"])
                axes[ax].plot(np.percentile(v_each, q=50, axis=0), label='baseline', linestyle='--')
                axes[ax].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                    y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                    y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)

            axes[ax].grid(True, alpha=0.1)
            if env == '8500':
                axes[ax].title.set_text('{} node'.format(env))
            else:
                axes[ax].title.set_text('{} bus'.format(env))
            if ax == 0:
                axes[ax].set_ylabel(ylabels[metric])
            if ax == 1:
                axes[ax].set_xlabel(xlabel)
        plt.legend()
        path = Path('./res/figs/state_{}/reward_{}/{}_{}.pdf'.format(state_option,
                                                                     reward_option,
                                                                     envs,
                                                                     metric))
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight')
        print(path)
    
    # Separate block for relative performance
    for metric in ylabels.keys():
        if "_baseline" in metric:
            continue

        fig_rel, axes_rel = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.), tight_layout=True)
        valid = False
        for ax, env in enumerate(envs):
            baseline = np.array(results[env][algos[0]][metric+"_baseline"]) if metric+"_baseline" in results[env][algos[0]].keys() else None

            if baseline is not None:
                valid = True
                v_all = []
                for res in results[env].values():
                    if res[metric]:
                        v_all.append(np.array(res[metric]) - baseline)
                    else:
                        v_all.append([])

                for i, v_each in enumerate(v_all):
                    if isinstance(v_each, np.ndarray):
                        axes_rel[ax].plot(np.percentile(v_each, q=50, axis=0), label=algos[i])
                        axes_rel[ax].fill_between(np.arange(v_each.shape[1]),
                                                np.percentile(v_each, q=10, axis=0),
                                                np.percentile(v_each, q=90, axis=0), alpha=0.4)

                axes_rel[ax].grid(True, alpha=0.1)
                axes_rel[ax].set_title('{} {}'.format(env, 'node' if env == '8500' else 'bus (relative)'))
                if ax == 0:
                    axes_rel[ax].set_ylabel(ylabels[metric])
                if ax == 1:
                    axes_rel[ax].set_xlabel(xlabel)
                
        if not valid:
            plt.close(fig_rel)
            continue
        plt.legend()
        path = Path('./res/figs/state_{}/reward_{}/{}_{}_relative.pdf'.format(state_option,
                                                                                reward_option,
                                                                                envs,
                                                                                metric))
        plt.savefig(path, bbox_inches='tight')
        print(path)

    
    # Plot action differences relative to baseline
    fig, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.), tight_layout=True)

    for ax, env in enumerate(envs):
        baseline_actions = np.array(results[env][algos[0]]['online_action_baseline'])
        if baseline_actions.size == 0:
            continue
        for a in algos:
            action_diff = np.linalg.norm(np.array(results[env][a]['online_action']) - baseline_actions, axis=2)
            if action_diff.size > 0:
                axes[ax].plot(np.percentile(action_diff, q=50, axis=0), label=a)
                axes[ax].fill_between(np.arange(action_diff.shape[1]),
                                    np.percentile(action_diff, q=10, axis=0),
                                    np.percentile(action_diff, q=90, axis=0), alpha=0.4)
        
        axes[ax].grid(True, alpha=0.1)
        axes[ax].set_title('{} {}'.format(env, 'node' if env == '8500' else 'bus (action diff)'))
        if ax == 0:
            axes[ax].set_ylabel('Action Difference')
        if ax == 1:
            axes[ax].set_xlabel(xlabel)

    plt.legend()
    path = Path('./res/figs/state_{}/reward_{}/{}_action_difference.pdf'.format(state_option,
                                                                                reward_option,
                                                                                envs))
    plt.savefig(path, bbox_inches='tight')
    print(path)


    # Plot action step differences relative to the previous step
    fig, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.), tight_layout=True)

    for ax, env in enumerate(envs):
        baseline_actions = np.array(results[env][algos[0]]['online_action_baseline'])
        baseline_diff = np.linalg.norm(baseline_actions[:, 1:, :] - baseline_actions[:, :-1, :], axis=2)

        for a in algos:
            actions = np.array(results[env][a]['online_action'])
            action_diff = np.linalg.norm(actions[:, 1:, :] - actions[:, :-1, :], axis=2)
            if action_diff.size > 0:
                axes[ax].plot(np.percentile(action_diff, q=50, axis=0), label=f'{a} diff')
                axes[ax].fill_between(np.arange(action_diff.shape[1]),
                                    np.percentile(action_diff, q=10, axis=0),
                                    np.percentile(action_diff, q=90, axis=0), alpha=0.4)
        
        if baseline_diff.size > 0:
            axes[ax].plot(np.percentile(baseline_diff, q=50, axis=0), label='baseline diff', linestyle='--')
            axes[ax].fill_between(np.arange(baseline_diff.shape[1]),
                                np.percentile(baseline_diff, q=10, axis=0),
                                np.percentile(baseline_diff, q=90, axis=0), alpha=0.4)
            
        axes[ax].grid(True, alpha=0.1)
        axes[ax].set_title('{} {}'.format(env, 'node' if env == '8500' else 'bus (step diff)'))
        if ax == 0:
            axes[ax].set_ylabel('Action Step Difference')
        if ax == 1:
            axes[ax].set_xlabel(xlabel)

    plt.legend()
    path = Path('./res/figs/state_{}/reward_{}/{}_action_step_difference.pdf'.format(state_option,
                                                                                reward_option,
                                                                                envs))
    plt.savefig(path)
    print(path)

    fig, axes = plt.subplots(nrows=1, ncols=len(envs), figsize=(4.*len(envs), 4.), tight_layout=True)
    for ax, env in enumerate(envs):
        for a in algos:
            rewards = np.array(results[env][a]['online_reward'])
            if rewards.size > 0:
                cum_rewards = np.cumsum(rewards, axis=1)
                axes[ax].plot(np.percentile(cum_rewards, q=50, axis=0), label=a)
                axes[ax].fill_between(np.arange(cum_rewards.shape[1]),
                                    np.percentile(cum_rewards, q=10, axis=0),
                                    np.percentile(cum_rewards, q=90, axis=0), alpha=0.4)
            
            baseline_rewards = np.array(results[env][a]['online_reward_baseline'])
            cum_baseline_rewards = np.cumsum(baseline_rewards, axis=1)
            if cum_baseline_rewards.size > 0:
                axes[ax].plot(np.percentile(cum_baseline_rewards, q=50, axis=0), label='baseline', linestyle='--')
                axes[ax].fill_between(np.arange(cum_baseline_rewards.shape[1]), 
                                    np.percentile(cum_baseline_rewards, q=10, axis=0),
                                    np.percentile(cum_baseline_rewards, q=90, axis=0), alpha=0.4)

        axes[ax].grid(True, alpha=0.1)
        axes[ax].set_title('{} {}'.format(env, 'node' if env == '8500' else 'bus'))
        if ax == 0:
            axes[ax].set_ylabel('Cumulative Reward')
        if ax == 1:
            axes[ax].set_xlabel(xlabel)

    plt.legend()
    path = Path('./res/figs/state_{}/reward_{}/{}_cumulative_rewards.pdf'.format(state_option,
                                                                                reward_option,
                                                                                envs))
    plt.savefig(path)
    print(path)
