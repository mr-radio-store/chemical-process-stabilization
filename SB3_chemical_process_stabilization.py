# ==========================================================
# Chemical Process Stabilization with SB3 (Single File)
# RPi5-safe | Headless plotting | Dynamic process
# ==========================================================

import gym
import numpy as np
from gym import spaces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# ==========================================================
# 1. Environment
# ==========================================================
class ChemicalStabilizationEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Time
        self.dt = 0.1
        self.max_steps = 500

        # Temperature
        self.T_env = 300.0
        self.setpoint = 350.0
        self.T_min = 280.0
        self.T_max = 420.0

        # Nominal parameters
        self.k_loss_nom = 0.05
        self.k_heat_nom = 10.0

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.T = 330.0 + 2.0 * np.random.randn()
        self.prev_u = 0.0
        self.step_count = 0

        # Parameter drift
        self.k_loss = self.k_loss_nom * np.random.uniform(0.8, 1.2)
        self.k_heat = self.k_heat_nom * np.random.uniform(0.8, 1.2)

        self.disturbance = 0.0

        return self._get_obs()

    def step(self, action):
        u = float(np.clip(action[0], 0.0, 1.0))

        # Disturbance update
        if self.step_count % 50 == 0:
            self.disturbance = np.random.uniform(-3.0, 3.0)

        # Process dynamics
        dT = (
            -self.k_loss * (self.T - self.T_env)
            + self.k_heat * u
            + self.disturbance
        )
        self.T += dT * self.dt

        error = self.T - self.setpoint

        # Reward
        reward = -error**2 - 0.02 * (u - self.prev_u)**2

        done = False
        if self.T < self.T_min or self.T > self.T_max:
            reward -= 500.0
            done = True

        self.prev_u = u
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([
            self.T,
            self.T - self.setpoint,
            self.prev_u,
            self.disturbance
        ], dtype=np.float32)


# ==========================================================
# 2. Train PPO
# ==========================================================
def train_model():
    env = ChemicalStabilizationEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_chemical_stabilizer")

    return model, env


# ==========================================================
# 3. Run & Save Figure
# ==========================================================
def evaluate_and_save(env, model, filename="stabilization_result.png"):
    obs = env.reset()
    T_log, u_log = [], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        T_log.append(obs[0])
        u_log.append(obs[2])

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(T_log, label="Temperature (K)")
    ax1.axhline(env.setpoint, linestyle="--", label="Setpoint")
    ax1.set_ylabel("Temperature (K)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(u_log, linestyle=":", label="Heater Input")
    ax2.set_ylabel("Control Input")

    plt.title("Chemical Process Stabilization (RL)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved figure: {filename}")


# ==========================================================
# 4. Main
# ==========================================================
if __name__ == "__main__":
    model, env = train_model()
    evaluate_and_save(env, model)
