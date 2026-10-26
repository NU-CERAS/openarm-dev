# OpenArm 参数化 skill 与音乐组合任务

日期：2026-10-24。

## 研究问题

在相同参数化 RL skill 库下，能够读取执行状态并调整工具调用的 VLM，能否提高 OpenArm 在未见过的乐器布局、skill 顺序和带时间约束任务中的完整任务成功率？

## 假设

参数化 skill 提供位置、姿态、物体和击打时间等目标输入，执行程序提供可观察的完成条件和共同时间基准，VLM 根据当前任务状态选择与组合 skill。这种组合有望支持训练任务以外的指令和顺序；效果需要对照实验验证。

## 当前证据

- 官方 `enactic/openarm_mjlab` 提供 MuJoCo Warp/PPO 的 reach、lift、pick-place 等任务源码。
- 官方 `enactic/openarm_isaac_lab` 提供 reach、lift 和 drawer 的训练配置。
- RoboPianist 提供 MuJoCo 钢琴模型、按键状态与 MIDI 事件代码。
- MarimbaBot 提供马林巴琴与琴槌模型，以及 Gazebo 击打检测和 MIDI 接口。
- SayCan 提供语言模型结合低层 skill 与可执行性估计的相关研究依据。
- 具体 commit、源码文件和证据边界见 [资源调研](openarm-rl-and-music.md)。

## 尚需验证

- 实际 OpenArm 的版本、gripper 形式、传感器和控制接口。
- goal-conditioned skill 在评估范围内的成功率与控制误差。
- 一项 skill 的结束状态能否支持下一项 skill 正常启动。
- 不依赖额外仿真 observation 的 actor 能否保持性能。
- 乐器模型在 MuJoCo 中的接触、目标判定与时间事件是否正确。
- 执行失败、物体移动和感知误差下的闭环恢复能力。
- 将单项 skill 的可靠性与完整任务规划的影响分别测量。

## 支持条件

使用相同 checkpoint、任务实例、观测来源和评估数量，闭环 VLM 条件在未见组合上的完整任务成功率高于一次任务规划条件，并报告置信区间。同时报告时间误差和额外击打次数，确认音乐约束满足程度。

## 否证条件

在匹配条件下，闭环决策没有改善完整任务指标；或收益主要来自额外的 simulator observation、不同 checkpoint、不同试验范围或不同成功条件。出现这些结果时，当前假设缺乏支持。

## 可执行的验证工作

核实真实硬件型号和控制接口，选择对应官方模型；复现官方 reach 与 pick-place 的训练和独立评估；建立能够从连续执行状态启动的 goal-conditioned skill；使用授权清楚的乐器资源建立接触与时间事件检查；在固定 skill 库下比较预定序列、一次 VLM 规划与闭环 VLM 决策。

本文记录研究问题与实验条件。训练结果、泛化结果和 sim2real 效果均尚未验证。
