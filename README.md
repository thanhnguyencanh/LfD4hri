# Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation

### V1.0, December 22th, 2025
**Authors:** [Thanh Nguyen Canh](https://thanhnguyencanh.github.io/), Thanh Tuan Tran, [Xiem HoangVan](https://sites.google.com/site/xiemhoang/), [Nak Young Chong](https://www.jaist.ac.jp/robot/).


LfD4hri is a novel “Human-to-Robot” imitation learning pipeline that enables robots to acquire manipulation skills directly from unstruc- tured video demonstrations, inspired by the human ability to learn by “watching” and “imitating”



# 1. License


If you use IRAF-SLAM in an academic work, please cite:
  
    @article{canh2026human,
      title={Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation},
      author={Thanh Nguyen Canh, Thanh Tuan Tran, Xiem HoangVan, and Nak Young Chong},
      journal={}, 
      volume={},
      number={},
      pages={},
      year={2025}
     }

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

## Reinforcement learning
Download...

# 3. Building and examples
