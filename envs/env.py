import numpy as np
from numpy import genfromtxt
from typing import Dict, Union, Any
import cmath
import opendssdirect as dss
from collections import defaultdict
from .env_utils import load_info
import gymnasium as gym
from gymnasium.envs.registration import register
import random


class VVCEnv(gym.Env):
    def __init__(self, options: Dict[str, Any] = {}):
        self.env_name = str(options["env"])
        self.state_option = str(options["state_option"])
        self.reward_option = str(options["reward_option"])
        self.discrete_action = options.get("discrete_action", False)

        self.len_total = 27648
        self.eval_offset = 20000
        self.train_offset = 0
        self.max_episode_length = 500

        # circuit model & raw ami data
        if self.env_name == "13":
            dss.run_command("Redirect envs/dss_13/IEEE13Nodeckt.dss")
            self.basekVA = 5000.0
            self.ami_data = genfromtxt(
                "./data/processed/first_2897_ami_aggto_580.csv",
                delimiter=",",
                max_rows=self.len_total,
            )
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 2.0)
            # the 13 bus feeder is highly loaded, an extra 2.0 is divided
        elif self.env_name == "123":
            dss.run_command("Redirect envs/dss_123/IEEE123Master.dss")
            self.basekVA = 5000.0
            self.ami_data = genfromtxt(
                "./data/processed/first_2897_ami_aggto_580.csv",
                delimiter=",",
                max_rows=self.len_total,
            )
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 1.0)
        elif self.env_name == "8500":
            dss.run_command("Redirect envs/dss_8500/Master.dss")
            self.basekVA = 27.5 * 1000
            self.ami_data = genfromtxt(
                "./data/processed/first_2897_ami.csv",
                delimiter=",",
                max_rows=self.len_total,
            )
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 2.0)

        # load info
        self.load_base, self.load_node = load_info(self.env_name)

        # vvc devices
        self.reg_names = dss.RegControls.AllNames()  # type: ignore
        self.cap_names = dss.Capacitors.AllNames()  # type: ignore

        # vvc data (offline & online)
        self.loss = genfromtxt(
            "./data/processed/{}/loss.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )[:, None]
        self.substation_pq = genfromtxt(
            "./data/processed/{}/substation_pq.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )
        self.load = genfromtxt(
            "./data/processed/{}/load.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )
        self.volt = genfromtxt(
            "./data/processed/{}/volt.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )
        self.ltc_tap = genfromtxt(
            "./data/processed/{}/tap.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )
        assert (
            len(self.load) == self.len_total
        ), f"len(self.load) != self.len_total, {len(self.load)} != {self.len_total}"
        if len(self.ltc_tap.shape) == 1:
            self.ltc_tap = self.ltc_tap[:, None]
        self.cap_status = genfromtxt(
            "./data/processed/{}/status.csv".format(self.env_name),
            delimiter=",",
            max_rows=self.len_total,
        )

        self.load_avg = np.average(self.load, axis=0)

        # RL info
        self.dims_time = (
            168 * 2,
            24 * 2,
        )  # period of time. e.g. for weekly pattern use 168 * 2 for half-hourly data
        self.dims_ltc = (33,) * len(self.reg_names)
        self.dims_cap = (2,) * len(self.cap_names)
        self.dims_action = self.dims_ltc + self.dims_cap
        dims = len(self.reg_names) + len(self.cap_names)
        if self.discrete_action:
            print("Using discrete action space")
            self.action_space = gym.spaces.MultiDiscrete(
                [33] * len(self.reg_names) + [2] * len(self.cap_names)
            )
        else:
            print("Using continuous action space")
            self.action_space = gym.spaces.Box(
                low=np.array([0.0] * dims),
                high=np.array([32.0] * len(self.reg_names) + [1.0] * len(self.cap_names)),
                dtype=np.float32,
            )
        self.dim_substation_pq = self.substation_pq.shape[1]
        self.dim_load = self.load.shape[1]
        self.dim_volt = self.volt.shape[1]

        self.coef_switching = 0.1 # Original: 0.1
        self.coef_volt = 0.5
        if self.reward_option in ("1", "3"):
            self.coef_loss = 1.0
        elif self.reward_option in ("2", "4"):
            self.coef_loss = 0.0

        if self.state_option in ("1", "3"):
            self.dim_state = (
                self.dim_substation_pq
                + self.dim_load
                + len(self.dims_action)
                + 2 * len(self.dims_time)
            )
            # the 2 in the last term is for cos AND sin encoding of periodic variable
        elif self.state_option in ("2",):
            self.dim_state = (
                self.dim_substation_pq + len(self.dims_action) + 2 * len(self.dims_time)
            )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.dim_state,), dtype=np.float64
        )

        self.global_time = 0
        self.state = None
        self.action_prev = None
        self.eval_mode = options.get("eval", False)

    def reset(
        self, seed: Union[None, int] = None, options: Union[None, Dict[str, Any]] = None
    ):
        # if offline = True, set the global time to the beginning of the full dataset;
        # otherwise set the global time to the beginning of the online dataset
        if self.eval_mode:
            if seed is None:
                self.global_time = random.randint(
                    self.eval_offset, self.len_total - 500
                )
            else:
                self.global_time = self.eval_offset + (
                    (500 * seed) % (self.len_total - self.eval_offset - 500)
                )
            assert (
                self.global_time >= self.eval_offset
                and self.global_time < self.len_total - 500
            )
        else:
            if seed is None:
                self.global_time = random.randint(0, self.eval_offset - 500)
            else:
                self.global_time = (500 * seed) % (self.eval_offset - 500)
            assert self.global_time < self.eval_offset and self.global_time >= 0

        if self.state_option == "3":
            self.global_time = max(self.global_time, 48)

        if self.state_option == "1":
            self.state = np.concatenate(
                [
                    self.substation_pq[self.global_time - 1, :] / (self.basekVA / 3.0),
                    self.load[self.global_time, :] / self.load_avg,
                ]
            )
        elif self.state_option == "2":
            self.state = np.concatenate(
                [
                    self.substation_pq[self.global_time - 1, :] / (self.basekVA / 3.0),
                ]
            )
        elif self.state_option == "3":
            self.state = np.concatenate(
                [
                    self.substation_pq[self.global_time - 1, :] / (self.basekVA / 3.0),
                    self.load[self.global_time - 48, :] / self.load_avg,
                ]
            )
        self.state = np.concatenate(
            [
                self.state,
                self.ltc_tap[self.global_time - 1, :],
                self.cap_status[self.global_time - 1, :],
                np.array(
                    [
                        np.cos(2 * np.pi * (self.global_time / ii))
                        for ii in self.dims_time
                    ]
                ),
                np.array(
                    [
                        np.sin(2 * np.pi * (self.global_time / ii))
                        for ii in self.dims_time
                    ]
                ),
            ]
        )
        self.action_prev = np.concatenate(
            (
                self.ltc_tap[self.global_time - 1, :],
                self.cap_status[self.global_time - 1, :],
            )
        )
        return self.state, {}

    @staticmethod
    def tap_to_tappu(tap):
        # from [0, 32] to [0.9, 1.1]
        pu_per_ltc_tap = 5 / 8 / 100  # 5/8 % voltage rule
        tap_pu = 1.0 + (tap - 16) * pu_per_ltc_tap
        return tap_pu

    @staticmethod
    def tappu_to_tap(tap_pu):
        # from [0.9, 1.1] to [0, 32]
        pu_per_ltc_tap = 5 / 8 / 100
        tap = (tap_pu - 1.0) / pu_per_ltc_tap + 16
        return tap

    @staticmethod
    def average_every_n(nparr, n):
        # given 1-D np array and an int n, return array([np.average(nparr[:n]), np.average(nparr[n:2*n]), ...])
        res = []
        for i in range(len(nparr) // n):
            res.append(np.average(nparr[i * n : (i + 1) * n]))
        res.append(np.average(nparr[len(nparr) // n * n :]))
        return np.array(res)

    def step(self, action=None):
        info: Dict[str, object] = defaultdict(lambda: None)

        loss: np.ndarray = np.array([-1.0])
        volt_120: np.ndarray = np.array([-1.0])
        volt_pu: np.ndarray = np.array([-1.0])
        substation_pq: np.ndarray = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        reward: float = -1.0
        if action is None:
            # if no action is provided, use the dss default control logic
            action = np.concatenate(
                [
                    self.ltc_tap[self.global_time, :],
                    self.cap_status[self.global_time, :],
                ]
            )
            volt_120 = self.volt[self.global_time, :]
            volt_pu = volt_120 / 120.0
            loss = self.loss[self.global_time, 0]
            substation_pq = self.substation_pq[self.global_time, :]
            info["PF converge"] = True
        else:
            # set load
            if self.env_name == "8500":
                load_kw = [
                    val * self.ami_data[self.global_time, i]
                    for i, (key, val) in enumerate(self.load_base.items())
                ]
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command("Load.{}.kW={}".format(key, load_kw[i]))
            elif self.env_name in ("13", "123"):
                load_kw = [
                    val[0] * self.ami_data[self.global_time, i]
                    for i, (key, val) in enumerate(self.load_base.items())
                ]
                load_kvar = [
                    val[1] * self.ami_data[self.global_time, i]
                    for i, (key, val) in enumerate(self.load_base.items())
                ]
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command("Load.{}.kW={}".format(key, load_kw[i]))
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command("Load.{}.kvar={}".format(key, load_kvar[i]))

            # set ltc tap and cap status (OpenDSS manual. pp-117)
            for i, reg in enumerate(self.reg_names):
                tap = self.tap_to_tappu(action[i])
                dss.run_command("transformer.{}.Taps=[1.0 {}]".format(reg, tap))
            for i, cap in enumerate(self.cap_names):
                dss.run_command(
                    "Capacitor.{}.status={}".format(
                        cap, action[i + len(self.reg_names)]
                    )
                )

            # solve
            dss.run_command("Set Controlmode=OFF")
            dss.Solution.Solve()  # type: ignore

            if not dss.Solution.Converged():  # type: ignore
                info["PF converge"] = False
            else:
                # voltage profile
                # ref.1: https://sourceforge.net/p/electricdss/discussion/861977/thread/a53badb5/
                # ref.2: https://github.com/dss-extensions/OpenDSSDirect.py/issues/15
                if self.env_name == "8500":
                    volt = []
                    for bus in self.load_node.keys():
                        dss.Circuit.SetActiveBus(bus)  # type: ignore
                        v_mag_angle = dss.Bus.puVmagAngle()  # type: ignore
                        v = v_mag_angle[::2]
                        ang = v_mag_angle[1::2]
                        v1 = cmath.rect(v[0], ang[0] / 180.0 * np.pi)
                        v2 = cmath.rect(v[1], ang[1] / 180.0 * np.pi)
                        v0 = abs(v1 - v2) / 2.0
                        volt.append(v0)  # smart meter measures phase-phase volt
                    volt_120 = np.around(np.array(volt) * 120.0, decimals=1)
                    volt_120 = self.average_every_n(volt_120, 10)
                    volt_pu = volt_120 / 120.0
                elif self.env_name in ("13", "123"):
                    volt = []
                    for bus in self.load_node.keys():
                        dss.Circuit.SetActiveBus(bus)  # type: ignore
                        v_mag_angle = dss.Bus.puVmagAngle()  # type: ignore
                        v = v_mag_angle[::2]
                        if self.load_node[bus]:
                            for i in range(len(self.load_node[bus])):
                                volt.append(v[i])
                        else:
                            volt.append(v[0])
                    volt_120 = np.around(np.array(volt) * 120.0, decimals=1)
                    volt_pu = volt_120 / 120.0

                # loss (kw)
                loss = np.around(np.array(dss.Circuit.Losses()[0] / 1000.0), decimals=1)  # type: ignore

                # total power (kw, kvar)
                # ref.1: https://sourceforge.net/p/electricdss/discussion/beginners/thread/6d771703/#0344
                dss.Circuit.SetActiveElement("Vsource.source")  # type: ignore
                substation_pq = dss.CktElement.Powers()[:6]  # type: ignore
                substation_p = substation_pq[::2]
                substation_q = substation_pq[1::2]
                substation_pq = np.around(
                    -np.array(substation_p + substation_q), decimals=1
                )

                info["PF converge"] = True

        # reward
        if info["PF converge"]:
            if self.reward_option in ("1", "2"):
                reward = -(
                    np.sum(np.round(np.abs(action - self.action_prev)))
                    * self.coef_switching
                    + np.sum(np.abs(volt_pu - 1.0)) * self.coef_volt
                    + loss / self.basekVA * self.coef_loss
                )
            elif self.reward_option in ("3", "4"):
                reward = -(
                    np.sum(np.round(np.abs(action - self.action_prev)))
                    * self.coef_switching
                    + np.sum(
                        np.logical_or(volt_pu < 0.95, volt_pu > 1.05).astype(float)
                    )
                    * self.coef_volt
                    + loss / self.basekVA * self.coef_loss
                )
        else:
            if self.reward_option in ("1", "2"):
                reward = (
                    -(
                        np.sum(np.round(np.abs(action - self.action_prev)))
                        * self.coef_switching
                        + 0.05 * self.dim_volt * self.coef_volt
                        + 1.0 * self.coef_loss
                    )
                    * 10.0
                )
            elif self.reward_option in ("3", "4"):
                reward = (
                    -(
                        np.sum(np.round(np.abs(action - self.action_prev)))
                        * self.coef_switching
                        + 1.0 * self.dim_volt * self.coef_volt
                        + 1.0 * self.coef_loss
                    )
                    * 10.0
                )

        # baseline reward (reward under dss policy)
        baseline_action = np.concatenate(
            [self.ltc_tap[self.global_time, :], self.cap_status[self.global_time, :]]
        )
        baseline_volt_120 = self.volt[self.global_time, :]
        baseline_volt_pu = baseline_volt_120 / 120.0
        baseline_loss = self.loss[self.global_time, 0]
        if self.reward_option in ("1", "2"):
            info["baseline_reward"] = -(
                np.sum(np.round(np.abs(baseline_action - self.action_prev)))
                * self.coef_switching
                + np.sum(np.abs(baseline_volt_pu - 1.0)) * self.coef_volt
                + baseline_loss / self.basekVA * self.coef_loss
            )
        elif self.reward_option in ("3", "4"):
            info["baseline_reward"] = -(
                np.sum(np.round(np.abs(baseline_action - self.action_prev)))
                * self.coef_switching
                + np.sum(
                    np.logical_or(baseline_volt_pu < 0.95, volt_pu > 1.05).astype(float)
                )
                * self.coef_volt
                + baseline_loss / self.basekVA * self.coef_loss
            )
        self.relative_reward = reward - info["baseline_reward"]

        # next state
        self.global_time += 1

        if not info["PF converge"]:
            substation_pq = self.substation_pq[self.global_time - 1, :]
        if self.state_option == "1":
            self.state = np.concatenate(
                [
                    substation_pq / (self.basekVA / 3.0),
                    self.load[self.global_time, :] / self.load_avg,
                ]
            )
        elif self.state_option == "2":
            self.state = np.concatenate(
                [
                    substation_pq / (self.basekVA / 3.0),
                ]
            )
        elif self.state_option == "3":
            self.state = np.concatenate(
                [
                    substation_pq / (self.basekVA / 3.0),
                    self.load[self.global_time - 48, :] / self.load_avg,
                ]
            )
        self.state = np.concatenate(
            [
                self.state,
                action,
                np.array(
                    [
                        np.cos(2 * np.pi * (self.global_time / ii))
                        for ii in self.dims_time
                    ]
                ),
                np.array(
                    [
                        np.sin(2 * np.pi * (self.global_time / ii))
                        for ii in self.dims_time
                    ]
                ),
            ]
        )

        if info["PF converge"]:
            info["loss"] = loss
            info["v"] = volt_120
            info["substation_pq"] = substation_pq
        info["switching_steps"] = np.sum(np.round(np.abs(action - self.action_prev)))
        info["action"] = action

        self.action_prev = action.copy()

        if self.global_time == self.len_total - 1:
            done = True
        else:
            done = False
        return self.state, reward, done, False, info


register(
    id="VCCEnv-0",
    entry_point=VVCEnv,
    max_episode_steps=500,
)

if __name__ == "__main__":
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.env_checker import check_env
    import os

    env_config = {
        "env": "13",
        "state_option": 2,
        "reward_option": 1,
        "path": os.getcwd(),
    }

    # Create environment
    env = gym.make("VCCEnv-0", options=env_config)
    check_env(env)

    # Instantiate the agent
    model = SAC("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
