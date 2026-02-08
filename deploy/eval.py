import numpy as np
from datetime import datetime
import gymnasium as gym
import env_eval
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from models.td3 import Agent
from utils.normalizer import RunningStatNormalizer
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="td3")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000, type=int)  # purely random action
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # checkpoint log path
    parser.add_argument("--max_episode", default=1, type=int)  # Max episode to run environment for
    parser.add_argument("--max_steps", default=500, type=int)  # max steps per episode
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_args()

    file_name = "%s_%s" % (args.policy_name, args.env_name)
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    # Initialize saving file
    base = f'{args.policy_name}_{str(args.action)}'
    os.makedirs(os.path.join(base, args.log_path), exist_ok=True)

    writer = SummaryWriter(os.path.join(base, args.log_path))

    # Create environment (connects to real robot automatically)
    print("---------------------------------------")
    print("Creating environment and connecting to robot...")
    print("---------------------------------------")
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps
    env.unwrapped.max_episode = args.max_episode
    env.unwrapped.writer = writer

    normalizer = RunningStatNormalizer(shape=(12,))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, warmup=args.warmup, lr=1e-4,
                   writer=writer, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                   noise_clip=args.noise_clip, policy_freq=args.policy_freq, normalizer=normalizer,
                   chkpt_dir=os.path.join(base, args.checkpoint))
    policy.load_best()

    # Debug: Print normalizer stats
    print(f"Normalizer stats - n: {normalizer.n}, mean: {normalizer.mean}, std: {normalizer.std}")

    env.unwrapped.action_type = args.action
    curr_episode = 0
    success_rate = 0
    t0 = time.time()

    print("---------------------------------------")
    print("Starting evaluation on real robot...")
    print("---------------------------------------")

    try:
        while curr_episode < args.max_episode:
            n_steps = 0
            current_datetime = datetime.now()
            timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
            done = False

            print(f"\n=== Episode {curr_episode + 1}/{args.max_episode} ===")
            state, _ = env.reset()

            while not done:
                # Normalize state for policy
                normalized_state = state.copy()
                normalized_state[:12] = normalizer.normalize(state[:12])

                # Debug: Print raw vs normalized state (first step only)
                if n_steps == 0:
                    print(f"Raw state[:12]: {state[:12]}")
                    print(f"Normalized state[:12]: {normalized_state[:12]}")
                    print(f"Task state (state[14:30]): {state[14:30]}")

                # Get action from policy
                action = policy.choose_action(normalized_state, validation=True)

                # Execute on real robot (handled by environment)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                n_steps += 1

            # Track success
            is_success = any(info["log"])
            success_rate += is_success
            print(f"Episode {curr_episode + 1}: {'SUCCESS' if is_success else 'FAILED'} ({n_steps} steps)")

            curr_episode += 1

        success_rate /= args.max_episode
        elapsed = time.time() - t0

        print("\n---------------------------------------")
        print(f"Evaluation complete!")
        print(f"Episodes: {args.max_episode}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print("---------------------------------------")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")

    finally:
        # Cleanup (environment handles robot cleanup)
        print("Cleaning up...")
        env.close()
        print("Done")
