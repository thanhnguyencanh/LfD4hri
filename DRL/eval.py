import numpy as np
from datetime import datetime
import gym
import env_eval
from moviepy.editor import ImageSequenceClip
import argparse
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time
from utils.relay_buffer import ReplayBuffer
from models.td3 import Agent
from utils.normalizer import RunningStatNormalizer
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="td3")  # Policy name
    parser.add_argument("--env_name", default="ImitationLearning-v1")  # environment name
    parser.add_argument("--warmup", default=20000, type=int)  # purely random action
    parser.add_argument("--action", default=0, type=int, required=True)  # training action
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_path", default='reward_log', type=str)  # reward log path
    parser.add_argument("--checkpoint", default='ckpt', type=str)  # checkpoint log path
    parser.add_argument("--video", default='demo', type=str)  # demo path
    parser.add_argument("--max_episode", default=50, type=int)  # Max episode to run environment for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # Evaluate frequency
    parser.add_argument("--expl_noise", default=0.2, type=float)  # Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic (recommend 256)
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--max_steps", default=100, type=int)  # max steps per episode
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
    for path in [args.log_path, args.video]:
        path = os.path.join(base, path)
        # if os.path.isdir(path):
        #     shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(os.path.join(base, args.log_path))
    env = gym.make(args.env_name)
    env.unwrapped.max_steps = args.max_steps  # define max steps
    env.unwrapped.max_episode = args.max_episode  # define max steps
    normalizer = RunningStatNormalizer(shape=(12,))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action, warmup=args.warmup, lr=1e-4,
                   writer=writer, discount=args.discount, tau=args.tau, policy_noise=args.policy_noise,
                   noise_clip=args.noise_clip, policy_freq=args.policy_freq, normalizer=normalizer, chkpt_dir=os.path.join(base, args.checkpoint))
    policy.load()
    replay_buffer = ReplayBuffer()
    env.unwrapped.action_type = args.action
    env.unwrapped.curriculum_learning = 50000
    env.unwrapped.writer = writer
    curr_episode = 0
    episode_reward = 0
    success = []
    t0 = time.time()
    env.unwrapped.virtualize = True
    env.unwrapped.eval = True
    success_rate = 0
    '''TRAINING PROCESS'''
    while curr_episode < args.max_episode:
        n_steps = 0
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
        done = False
        state = env.reset()
        while not done:
            normalized_state = state.copy()
            normalized_state[:12] = normalizer.normalize(state[:12])
            action = policy.choose_action(normalized_state, validation=True)
            state, reward, done, info = env.step(action)
            n_steps += 1
        success_rate += any(info["log"])
        is_success = any(info["log"])
        if is_success:
            print("...video saving...")
            debug_clip = ImageSequenceClip(env.cache_front_video, fps=15)
            video_path = os.path.join(os.path.join(base, args.video), f"demo_ep{curr_episode}_view_0_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)

            debug_clip = ImageSequenceClip(env.cache_side_video, fps=15)
            video_path = os.path.join(os.path.join(base, args.video), f"demo_ep{curr_episode}_view_1_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)

            debug_clip = ImageSequenceClip(env.cache_diagonal_video, fps=15)
            video_path = os.path.join(os.path.join(base, args.video), f"demo_ep{curr_episode}_view_2_action_{env.unwrapped.action_type}_{timestamp}_state_{is_success}.mp4")
            debug_clip.write_videofile(video_path, fps=15)
            print(f"Video saved to {video_path}")

        curr_episode += 1
    success_rate /= args.max_episode

    print("---------------------------------------")
    print(f"Evaluation over {args.max_episode} episodes: {success_rate:.3f}")
    print("---------------------------------------")
