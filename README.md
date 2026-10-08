# berth_scheduling_optimizer
港口泊位智能调度系统：基于开源大模型与MILP的智能调度顾问，支持连续泊位与岸桥联动。
# 港口泊位智能调度系统
基于开源大模型与混合整数线性规划（MILP）的港口泊位调度顾问。
## 功能特性
- 支持泊位分配、岸桥分配、作业时序与连续岸线空间约束
- 支持优先级、依赖关系、靠泊限制、潮汐窗口、最晚离港等约束
- 无可行解时输出诊断报告，不编造方案
- 提供 Flask API 接口，可集成至智能体应用
## 运行环境
Python 3.9+，依赖 Flask、PuLP、Matplotlib
## 运行方法
```bash
pip install -r requirements.txt
python berth_scheduler.py
