# Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation

### V1.0, December 22th, 2025
**Authors:** [Thanh Nguyen Canh](https://thanhnguyencanh.github.io/), Thanh Tuan Tran, [Xiem HoangVan](https://sites.google.com/site/xiemhoang/), [Nak Young Chong](https://www.jaist.ac.jp/robot/).


LfD4hri is a novel “Human-to-Robot” imitation learning pipeline that enables robots to acquire manipulation skills directly from unstruc- tured video demonstrations, inspired by the human ability to learn by “watching” and “imitating”



## 1. Project Structure

    LfD4HRI/
    │
    ├── DRL/
    │   ├── assets/                 Objects and robot configuration files
    │   ├── env/                    Reinforcement learning environments
    │   ├── models/                 RL models (TD3, SAC, ...)
    │   ├── reward/                 Reward function definitions
    │   └── train_td3.py            Main RL training code
    │
    ├── LLaVA/                      VLMs for object identification
    ├── video_keyframes_detector/   Video keyframe extraction
    ├── mmaction2/                  Action recognition framework
    ├── utils/                      Utility functions (overlap, blur, ...)
    └── demo.py                     Video Understanding main code

# 2. Prerequisites
Install all the python dependencies for video understanding using pip:
```
pip install -r requirements.txt
```
Run following command to run reinforcement learning:
```
pip install -r DRL/requirements.txt
```
# Checkpoints
## Video understanding
Download both [action recognition checkpoint](https://drive.google.com/file/d/1oZpapQmfzchaC9-GR4uIrawlye-kXaVf/view?usp=drive_link) and [hand detection](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE">faster_rcnn_1_8_132028.pth) and save them as follows:

Action recognition checkpoint to `/Human-to-Robot-Interaction/mmaction2/pretrained_file_and_checkpoint`
Hand detection checkpoint to `/Human-to-Robot-Interaction/video_understanding_checkpoint/res101_handobj_100K/pascal_voc`

## Reinforcement Learning
The system supports two robot arms, **UR5e** and **UF850**, each with four discrete action modes (```0: reach, 1: pick, 2: move, 3: put```)
We recommend adjusting the success threshold—defined as the acceptable positional error by modifying `threshold_1` in the following configuration file:
### Training

To train a reinforcement learning policy, run:
```bash
cd DRL
python3 train_td3.py --action {0,1,2,3} --robot {ur5e,uf850}
```
Note: Select only one value for each argument from the options listed above.
Starting a new training will automatically reset all existing checkpoints, logs, and related training artifacts.

# 3. Building and examples


# 4. License

# 5. Citation

If you use this work in an academic work, please cite:
  
    @article{canh2026human,
      title={Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation},
      author={Thanh Nguyen Canh, Thanh Tuan Tran, Xiem HoangVan, and Nak Young Chong},
      journal={}, 
      volume={},
      number={},
      pages={},
      year={2025}
     }
