"""
泊位调度优化程序
使用 PuLP 建立整数规划模型，最小化船舶的总（加权）在港时间，
支持通过 JSON 传入潮汐窗口、依赖关系、优先级权重、靠泊限制、
最小间隔、自定义硬约束、连续岸线位置（CBAP，连续泊位调度）、
装卸量动态作业时长与全局岸桥总数约束等可选条件，
并通过 Flask 提供 HTTP API 接口。
"""

import os

from flask import Flask, request, jsonify
import pulp

# matplotlib 使用非交互式 Agg 后端，保证在无显示器的服务器/花生壳环境也能出图；
# 必须在 pyplot 导入前指定
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Windows 中文字体设置，避免图中船名等中文显示为方块
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

app = Flask(__name__)

# 甘特图默认保存路径：与本脚本同目录下的 berth_gantt.png
DEFAULT_GANTT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "berth_gantt.png")


def plot_gantt(result_ships, num_berths, output_path=DEFAULT_GANTT_PATH):
    """
    根据求解结果绘制泊位调度甘特图（横向条形图）并保存为 PNG。

    参数:
        result_ships: optimize_berth_scheduling 返回的 ships 列表，
            每项含 name, berth, start_time, end_time
        num_berths: 泊位数量（决定纵轴行数）
        output_path: PNG 保存路径，默认为脚本同目录 berth_gantt.png

    返回:
        (output_path, _): 图片本地绝对路径 与 占位空串
        （已改为不返回 Base64，以加快 API 响应；PNG 仍正常保存到本地）
    """
    fig, ax = plt.subplots(figsize=(10, max(3, 0.8 * num_berths + 1.5)))

    # 不同船舶使用不同颜色（泊位为纵轴，颜色仅用于区分条块）
    cmap = plt.get_cmap("tab10")

    # 纵向按泊位排列：barh 的 y 为泊位号，left 为开始时间，width 为作业时长
    for idx, ship in enumerate(result_ships):
        berth = ship["berth"]
        start = ship["start_time"]
        duration = ship["end_time"] - ship["start_time"]
        ax.barh(berth, duration, left=start, height=0.55,
                color=cmap(idx % 10), edgecolor="black", alpha=0.85)
        # 条块中央标注船名
        ax.text(start + duration / 2, berth, ship["name"],
                ha="center", va="center", fontsize=10, color="black")

    # 纵轴：泊位标签（展示为“泊位1、泊位2……”，泊位1在最上方）
    ax.set_yticks(range(num_berths))
    ax.set_yticklabels([f"泊位{b + 1}" for b in range(num_berths)])
    ax.invert_yaxis()

    # 横轴：默认展示 0~24 小时；若最大结束时间超过 24，则自动延伸
    max_end = max((ship["end_time"] for ship in result_ships), default=0)
    ax.set_xlim(0, max(24, max_end))

    ax.set_xlabel("时间（小时）")
    ax.set_ylabel("泊位")
    ax.set_title("泊位调度方案甘特图")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)  # 显式关闭，避免多次调用时内存累积

    # 不再读取 PNG 转 Base64（加快 API 响应），仅返回图片本地绝对路径
    return os.path.abspath(output_path), ""


def optimize_berth_scheduling(ships, num_berths,
                              tide_windows=None,
                              dependencies=None,
                              priority_weights=None,
                              berth_restrictions=None,
                              min_gap=0.0,
                              custom_constraints=None,
                              objective="weighted_time",
                              quay_length=1000,
                              total_cranes=None,
                              crane_step=0.5):
    """
    使用 PuLP 建立整数规划模型求解泊位调度问题
    （离散泊位 + 连续岸线 CBAP + 岸桥数决策的混合模型）。

    参数:
        ships: 船舶列表，每艘船为 dict：
            - 基础字段：name, arrival；作业时长二选一：
              a) service（固定作业时长，小时，原有方式）
              b) cargo + cargo_type + max_cranes（模块一，由装卸量动态计算：
                 集装箱 service_i = cargo_i/(n_i×30)，
                 散货   service_i = cargo_i/(n_i×400)，
                 n_i 为分配的岸桥数，1 <= n_i <= max_cranes）
            - 可选 length（船身长度，米，用于 CBAP 空间约束，缺省 0）
        num_berths: 可用泊位数量
        total_cranes: 港口总岸桥数（模块二；None 表示不启用全局岸桥约束）
        crane_step: 全局岸桥约束的时间离散化步长（小时，默认 0.5），
            步长越小约束越精确、求解越慢
        quay_length: 连续岸线总长 L（米），默认 1000；
            时间上重叠的船在岸线上空间不得重叠
        tide_windows: 各泊位的不可作业（潮汐）区间，形如
            [[[9,11],[15,17]], [], [[12,14]]]，外层下标为泊位号（0-based）
        dependencies: 依赖关系 dict，如 {"C": ["A"], "E": ["B","D"]}，
            表示右侧被依赖船全部离港后，该船才能开始作业
        priority_weights: 优先级权重 dict，如 {"A": 3}；未列出的船默认权重 1
        berth_restrictions: 靠泊限制 dict，如 {"A": [0,1]}（0-based 泊位索引），
            表示该船只能停靠列出的泊位
        min_gap: 同泊位前后两船之间的最小间隔（小时），默认 0（不启用）
        custom_constraints: 自定义硬约束列表，当前支持
            {"type": "deadline", "ship": "A", "time": 16.0}
        objective: 目标函数类型，"weighted_time"（默认，加权总在港时间）
            或 "total_time"（总在港时间，所有权重视为 1）

    返回:
        dict: 统一 JSON 骨架
            {
              "status": "success" / "infeasible",
              "schedule": 每艘船的泊位、起止时间、岸线位置（start_pos/end_pos，米）
                          与分得岸桥数 cranes（未启用模块一时为 null）,
              "total_waiting": 总等待时间,
              "weighted_total": 加权总在港时间（Σ 权重×在港，未给权重默认 1）,
              "crane_peak": 岸桥同时使用的峰值（模块二）,
              "gantt_image": 提示文字（"甘特图已生成，保存在本地：<路径>"）,
              "message": 附加信息（正常为空；若作业时长经装卸量公式
                          强制校验后发生过修正，在此说明）
            }
    """
    n = len(ships)
    if n == 0:
        return {"status": "success", "schedule": [], "total_waiting": 0,
                "weighted_total": 0, "crane_peak": 0, "gantt_image": "",
                "message": ""}

    arrivals = [sh["arrival"] for sh in ships]
    names = [sh["name"] for sh in ships]
    # 船身长度（米）：连续岸线（CBAP）空间约束使用；未提供默认 0（不占岸线）
    lengths = [float(sh.get("length", 0)) for sh in ships]
    # 船舶名 -> 下标 映射，用于解析按名字传入的依赖/限制/自定义约束
    name_to_idx = {name: i for i, name in enumerate(names)}

    # ====== 模块一：作业时长由装卸量动态计算 ======
    # 提供 cargo（装卸量）+ cargo_type（container/bulk）的船，作业时长不再是
    # 输入参数，而是随分配的岸桥数 n_i 变化：
    #   集装箱：service_i = cargo_i / (n_i × 30)（单岸桥 30 TEU/小时）
    #   散货：  service_i = cargo_i / (n_i × 400)（单岸桥 400 吨/小时）
    # 未提供 cargo 的船沿用输入的 service（原有行为，向后兼容）
    CRANE_RATE = {"container": 30, "bulk": 400}
    has_cargo = [False] * n
    cargos = [0.0] * n
    rates = [0.0] * n
    max_cranes_list = [1] * n
    services = [0.0] * n   # 固定作业时长（未提供 cargo 的船）
    svc_max = [0.0] * n    # 作业时长上界（n_i=1 时最长），用于 big-M 与时间网格
    for i, sh in enumerate(ships):
        if sh.get("cargo") is not None and sh.get("cargo_type") is not None:
            if sh["cargo_type"] not in CRANE_RATE:
                raise ValueError(
                    f"船舶 {sh.get('name')} 的 cargo_type 非法: "
                    f"{sh['cargo_type']}（仅支持 container/bulk）")
            has_cargo[i] = True
            cargos[i] = float(sh["cargo"])
            rates[i] = CRANE_RATE[sh["cargo_type"]]
            max_cranes_list[i] = int(sh.get("max_cranes", 1))
            svc_max[i] = cargos[i] / rates[i]
        else:
            if sh.get("service") is None:
                raise ValueError(
                    f"船舶 {sh.get('name')} 需提供 service 或 cargo+cargo_type")
            services[i] = float(sh["service"])
            svc_max[i] = services[i]

    # big-M 的取值：足够大以放松非绑定约束（时间维度）
    M = max(arrivals) + sum(svc_max) + 1
    # 连续岸线长度 L 与空间维度的 big-M（岸线长 + 最大船长足以放松所有空间约束）
    L = float(quay_length)
    M_pos = L + (max(lengths) if lengths else 0) + 1

    # 创建整数规划问题（目标函数在下方按 objective 参数设置）
    prob = pulp.LpProblem("Berth_Scheduling", pulp.LpMinimize)

    # ====== 决策变量 ======

    # x[i][b]: 二进制变量，船 i 是否分配到泊位 b
    x = [[pulp.LpVariable(f"x_{i}_{b}", cat="Binary") for b in range(num_berths)]
         for i in range(n)]

    # s[i]: 整数变量，船 i 的开始服务时间
    s = [pulp.LpVariable(f"s_{i}", lowBound=0, cat="Integer") for i in range(n)]

    # y[i][j]: 二进制变量，船 i 是否在船 j 之前开始服务
    # 用于 big-M 方法处理同一泊位上的服务不重叠约束
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(n)]
         for i in range(n)]

    # ====== 连续泊位（CBAP）决策变量 ======

    # p[i]: 连续变量，船 i 在岸线上的起始位置 start_pos_i（米，>=0；
    #       上界由约束 p_i + length_i <= L 保证）
    p = [pulp.LpVariable(f"pos_{i}", lowBound=0, cat="Continuous")
         for i in range(n)]

    # u[i][j]/v[i][j]: 时间先后指示（仅对 i<j 使用）
    #   u_ij=0 强制 i 在时间上先于 j（s_i+svc_i <= s_j）
    #   v_ij=0 强制 j 在时间上先于 i（s_j+svc_j <= s_i）
    #   u_ij=v_ij=1 表示允许 i、j 在时间上重叠
    u = [[pulp.LpVariable(f"u_{i}_{j}", cat="Binary") for j in range(n)]
         for i in range(n)]
    v = [[pulp.LpVariable(f"v_{i}_{j}", cat="Binary") for j in range(n)]
         for i in range(n)]

    # zsp[i][j]: 空间先后指示（仅对 i<j 使用），zsp_ij=1 表示 i 位于 j 左侧
    #   （start_pos_i + length_i <= start_pos_j）
    #   命名为 zsp 以避免与潮汐约束循环内的局部变量 z 冲突
    zsp = [[pulp.LpVariable(f"z_{i}_{j}", cat="Binary") for j in range(n)]
           for i in range(n)]

    # ====== 模块一决策变量：岸桥数选择 ======
    # w[i][k]: 二进制变量，船 i 恰好分配 k 台岸桥（仅对提供 cargo 的船创建）。
    # 作业时长 service = cargo/(n×rate) 关于 n 非线性，但 n 只能取有限整数
    # 1..max_cranes，故用选择变量枚举线性化：
    #   n_i = Σ_k k·w_ik，service_i = Σ_k (cargo_i/(k·rate_i))·w_ik，Σ_k w_ik = 1
    w = [None] * n
    cranes_expr = [None] * n   # n_i 的线性表达式
    svc_expr = [None] * n      # service_i 的线性表达式
    for i in range(n):
        if has_cargo[i]:
            mc = max_cranes_list[i]
            w[i] = {k: pulp.LpVariable(f"w_{i}_{k}", cat="Binary")
                    for k in range(1, mc + 1)}
            cranes_expr[i] = pulp.lpSum(k * w[i][k] for k in range(1, mc + 1))
            svc_expr[i] = pulp.lpSum((cargos[i] / (k * rates[i])) * w[i][k]
                                     for k in range(1, mc + 1))

    # 离港时间表达式 end_expr[i] = s_i + service_i：
    # 提供 cargo 的船 service 是随 w_ik 变化的表达式（模块一），
    # 其余船为常数，等价于原有的 s_i + services[i]
    end_expr = [s[i] + (svc_expr[i] if has_cargo[i] else services[i])
                for i in range(n)]

    # ====== 目标函数（可选模式） ======
    # 在港时间 = 离港 - 到达 = end_expr_i - arrival_i = 等待 + 作业时长。
    # 对固定作业时长的船，作业时长为常数，最小化加权在港与加权等待等价；
    # 对由装卸量动态计算时长的船（模块一），作业时长随岸桥数变化，
    # 必须完整计入在港时间，才能反映“多派岸桥→缩短作业→减少在港”的权衡。
    # - weighted_time（默认）：权重取 priority_weights 中该船的权重，未列出默认 1
    # - total_time：所有权重视为 1，即最小化总在港时间
    stay_time = [end_expr[i] - arrivals[i] for i in range(n)]
    # 每艘船的优先级权重：priority_weights 中给出则取给定值，否则默认 1
    weights = ([float(priority_weights.get(name, 1)) for name in names]
               if priority_weights else [1.0] * n)
    if objective == "weighted_time" and priority_weights:
        prob += pulp.lpSum(weights[i] * stay_time[i] for i in range(n))
    else:
        prob += pulp.lpSum(stay_time)

    # ====== 约束条件 ======

    # 约束 1：每艘船必须分配且仅分配一个泊位
    for i in range(n):
        prob += pulp.lpSum(x[i][b] for b in range(num_berths)) == 1, f"assign_berth_{i}"

    # 约束 2：开始服务时间不能早于到达时间
    for i in range(n):
        prob += s[i] >= arrivals[i], f"no_early_start_{i}"

    # 约束 3：同一泊位上任意两艘船的服务时间不能重叠（big-M 方法）
    # 对每对船 (i, j) 其中 i < j，以及每个泊位 b：
    #   若 i 和 j 都在泊位 b，且 i 在 j 之前服务 => s[i] + svc[i] + gap <= s[j]
    #   若 i 和 j 都在泊位 b，且 j 在 i 之前服务 => s[j] + svc[j] + gap <= s[i]
    # 使用 big-M 放松非同一泊位或相对顺序相反时的约束
    # 若传入 min_gap > 0（岸桥连续性间隔），则在同泊位前后两船之间追加最小间隔：
    #   数学含义：同泊位上紧邻作业的两船，后船开始时间 >= 前船离港时间 + min_gap
    min_gap_val = float(min_gap or 0)
    for i in range(n):
        for j in range(i + 1, n):
            for b in range(num_berths):
                # i 先于 j 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (end_expr[i] + min_gap_val <= s[j]
                         + M * (1 - y[i][j])
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_a"
                # j 先于 i 服务，或 i、j 不在同一泊位 b 时约束放松
                prob += (end_expr[j] + min_gap_val <= s[i]
                         + M * y[i][j]
                         + M * (2 - x[i][b] - x[j][b])), \
                    f"overlap_{i}_{j}_{b}_b"

    # 约束 4：潮汐约束 —— 分配到泊位 b 的船，作业不得落在该泊位的不可作业区间内
    # 输入 tide_windows[b] = [[t1,t2], ...]：泊位 b 的若干潮汐窗口 [t1, t2)
    # 数学含义：若船 i 分配到泊位 b（x[i][b]=1），其作业区间 [s_i, s_i+svc_i)
    #          必须整体避开每个窗口，即
    #          s_i + svc_i <= t1（窗口开始前完成）或 s_i >= t2（窗口结束后才开始）
    # 引入二进制变量 z 在两种情形中选择；x[i][b]=0 时两式均被 big-M 放松
    for b in range(num_berths):
        windows = tide_windows[b] if tide_windows and b < len(tide_windows) else []
        for k, window in enumerate(windows):
            t1, t2 = window
            for i in range(n):
                z = pulp.LpVariable(f"tide_{i}_{b}_{k}", cat="Binary")
                # 情形 A（z=0）：船在窗口开始前完成作业
                prob += (end_expr[i] <= t1
                         + M * (1 - x[i][b]) + M * z), \
                    f"tide_before_{i}_{b}_{k}"
                # 情形 B（z=1）：船在窗口结束后才开始作业
                prob += (s[i] >= t2
                         - M * (1 - x[i][b]) - M * (1 - z)), \
                    f"tide_after_{i}_{b}_{k}"

    # 约束 5：依赖关系 —— 被依赖船全部离港后，依赖它们的船才能开始作业
    # 输入 dependencies = {"C": ["A"], "E": ["B","D"], ...}
    # 数学含义：对每条依赖 j ← i：s_j >= s_i + svc_i
    #          （j 的开始时间不早于 i 的离港时间）
    for ship_name, prereq_names in (dependencies or {}).items():
        j = name_to_idx.get(ship_name)
        if j is None:
            continue  # 船名不在输入船舶列表中，忽略该条依赖
        for prereq_name in prereq_names:
            i = name_to_idx.get(prereq_name)
            if i is None:
                continue
            prob += s[j] >= end_expr[i], f"dependency_{i}_to_{j}"

    # 约束 6：靠泊限制 —— 某些船只能停靠指定泊位（0-based 泊位索引）
    # 输入 berth_restrictions = {"A": [0,1], ...}
    # 数学含义：对船 i 未被允许的泊位 b，强制 x[i][b] = 0
    for ship_name, allowed_berths in (berth_restrictions or {}).items():
        i = name_to_idx.get(ship_name)
        if i is None:
            continue
        allowed = set(allowed_berths)
        for b in range(num_berths):
            if b not in allowed:
                prob += x[i][b] == 0, f"berth_restriction_{i}_{b}"

    # 约束 7：自定义硬约束（当前支持 deadline 类型）
    # 输入 custom_constraints = [{"type": "deadline", "ship": "A", "time": 16.0}, ...]
    # deadline 数学含义：船必须在 time 之前离港，即 s_i + svc_i <= time
    for idx, cc in enumerate(custom_constraints or []):
        i = name_to_idx.get(cc.get("ship"))
        if cc.get("type") == "deadline" and i is not None:
            prob += end_expr[i] <= cc.get("time"), \
                f"custom_deadline_{i}_{idx}"

    # 约束 8：连续泊位（CBAP）—— 时间上重叠的船，在岸线上空间不得重叠
    # 输入：每艘船的 length（米，缺省 0）与岸线总长 quay_length（L，米）
    # 数学含义：把每艘船看作时空矩形 [s_i, s_i+svc_i) × [pos_i, pos_i+len_i)，
    #          任意两艘船的矩形不能同时在时间和空间两个维度上重叠，即下列
    #          四个条件至少成立其一：
    #          ① s_i+svc_i <= s_j（i 时间在前）   ② s_j+svc_j <= s_i（j 时间在前）
    #          ③ pos_i+len_i <= pos_j（i 空间在左）④ pos_j+len_j <= pos_i（j 空间在左）
    # 线性化：u_ij=0 强制①，v_ij=0 强制②；u=v=1（时间可能重叠）时，
    #         z_ij=1 强制③、z_ij=0 强制④；时间已分离时空间约束被 big-M 自动放松
    # 岸线边界：0 <= pos_i，且 pos_i + length_i <= L（船不能超出岸线）
    for i in range(n):
        prob += p[i] + lengths[i] <= L, f"pos_within_quay_{i}"

    for i in range(n):
        for j in range(i + 1, n):
            # 两船均不占岸线（长度均为 0）时不存在空间重叠，跳过以精简模型
            if lengths[i] <= 0 and lengths[j] <= 0:
                continue
            # 时间维：u_ij=0 => ①成立；v_ij=0 => ②成立（u=v=1 即允许时间重叠）
            prob += end_expr[i] <= s[j] + M * u[i][j], \
                f"time_seq_{i}_{j}_a"
            prob += end_expr[j] <= s[i] + M * v[i][j], \
                f"time_seq_{i}_{j}_b"
            # 空间维：仅当 u=v=1（时间可能重叠）时生效，zsp_ij 决定左右顺序
            # zsp_ij=1 => ③ start_pos_i + length_i <= start_pos_j
            prob += (p[i] + lengths[i] <= p[j]
                     + M_pos * (1 - zsp[i][j])
                     + M_pos * (2 - u[i][j] - v[i][j])), \
                f"space_seq_{i}_{j}_a"
            # zsp_ij=0 => ④ start_pos_j + length_j <= start_pos_i
            prob += (p[j] + lengths[j] <= p[i]
                     + M_pos * zsp[i][j]
                     + M_pos * (2 - u[i][j] - v[i][j])), \
                f"space_seq_{i}_{j}_b"

    # 约束 9：动态作业时长（模块一）—— 每艘提供 cargo 的船恰好选择一种岸桥数
    # 数学含义：Σ_k w_ik = 1，n_i = Σ_k k·w_ik ∈ [1, max_cranes_i]，
    #          service_i = Σ_k (cargo_i/(k·rate_i))·w_ik（rate：集装箱 30、散货 400）
    for i in range(n):
        if has_cargo[i]:
            prob += pulp.lpSum(w[i][k]
                               for k in range(1, max_cranes_list[i] + 1)) == 1, \
                f"crane_sel_{i}"

    # 约束 10：全局岸桥总数约束（模块二，时间离散化）
    # 输入 total_cranes（港口总岸桥数）与 crane_step（离散化步长，小时）
    # 数学含义：把时间轴按步长离散为网格点 t，要求每个网格点上
    #          Σ_i n_i·z_it <= total_cranes，其中 z_it=1 表示船 i 在时刻 t 正在作业。
    #          注：离散化只在采样点上保证约束，步长越小越精确（可用 crane_step 调节）
    # 占用变量 g_it（连续）表示船 i 在 t 时刻占用的岸桥数，目标是让
    # g_it = n_i·(正在作业)。作业区间为左闭右开 [s_i, c_i)，“正在作业”意为
    # s_i <= t < c_i；其否定为“未开工(s_i >= t+δ) 或 已离港(c_i <= t)”，
    # 是一个“或”结构，需要两个指示变量分别表达（单一 z 无法正确表示该析取）：
    #   na_it = 1 ⇒ 尚未开工：s_i >= t + δ        （big-M 放松形式）
    #   fi_it = 1 ⇒ 已经离港：c_i <= t            （big-M 放松形式）
    #   g_it <= n_i
    #   g_it >= n_i - M_cr·(na_it + fi_it)
    #     ⇒ 正在作业（na=fi=0）时 g_it = n_i；未作业时求解器自然取 g_it = 0
    if total_cranes is not None and any(has_cargo):
        step = float(crane_step or 0.5)
        delta = 1e-4
        # 时间网格上界：按“所有船 n=1 最长时长串行作业”的保守估算
        H = max(arrivals) + sum(svc_max) + (min_gap_val * n) + 2
        M_cr = H + 1  # 岸桥约束专用 big-M（需覆盖网格上界 H）
        grid = [round(step * t, 6) for t in range(int(H / step) + 1)]
        g = {}    # g[(i, ti)]: 船 i 在网格点 ti 占用的岸桥数
        na = {}   # na[(i, ti)]=1 表示船 i 在时刻 t 尚未开工
        fi = {}   # fi[(i, ti)]=1 表示船 i 在时刻 t 已经离港
        for i in range(n):
            if not has_cargo[i]:
                continue
            for ti, t in enumerate(grid):
                if t < arrivals[i] - 1e-9:
                    continue  # 到达时间之前不可能作业
                g[i, ti] = pulp.LpVariable(f"g_{i}_{ti}", lowBound=0,
                                           cat="Continuous")
                na[i, ti] = pulp.LpVariable(f"na_{i}_{ti}", cat="Binary")
                fi[i, ti] = pulp.LpVariable(f"fi_{i}_{ti}", cat="Binary")
                # na=1 ⇒ 尚未开工：s_i >= t + δ
                prob += s[i] >= t + delta - M_cr * (1 - na[i, ti]), \
                    f"crk_na_{i}_{ti}"
                # fi=1 ⇒ 已离港：c_i <= t
                prob += end_expr[i] <= t + M_cr * (1 - fi[i, ti]), \
                    f"crk_fi_{i}_{ti}"
                # 占用上界：g_it <= n_i
                prob += g[i, ti] <= cranes_expr[i], f"crk_g1_{i}_{ti}"
                # 正在作业（na=fi=0）⇒ g_it = n_i
                prob += g[i, ti] >= cranes_expr[i] - M_cr * (na[i, ti] + fi[i, ti]), \
                    f"crk_g2_{i}_{ti}"
        # 每个网格点的全局岸桥总量约束：Σ_i n_i·z_it <= total_cranes
        for ti in range(len(grid)):
            terms = [g[i, ti] for i in range(n) if (i, ti) in g]
            if terms:
                prob += pulp.lpSum(terms) <= total_cranes, f"crk_cap_{ti}"

    # ====== 求解 ======
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        # 模型无解（如 deadline 过早、岸桥不足、依赖成环等）时返回统一骨架，而非抛错崩溃
        return {"status": "infeasible", "schedule": [], "total_waiting": 0,
                "weighted_total": 0, "crane_peak": 0, "gantt_image": "",
                "message": "模型无可行解，请放宽潮汐窗口、靠泊限制、截止时间或岸桥数量等约束"}

    # ====== 提取结果（schedule 列表） ======
    schedule = []
    total_wait = 0
    weighted_total = 0.0
    corrections = []  # 强制校验中发现并修正的作业时长记录

    for i in range(n):
        start_time = int(pulp.value(s[i]))
        wait = start_time - arrivals[i]
        total_wait += wait

        # 找出船 i 被分配到哪个泊位
        berth_assigned = None
        for b in range(num_berths):
            if pulp.value(x[i][b]) > 0.5:
                berth_assigned = b
                break

        # 分配的岸桥数（模块一；未提供 cargo 的船为 None）
        cranes_used = (int(round(float(pulp.value(cranes_expr[i]))))
                       if has_cargo[i] else None)

        # ====== 强制校验：作业时长必须符合装卸量公式 ======
        # 集装箱：service = cargo / (cranes × 30)
        # 散货：  service = cargo / (cranes × 400)
        # 若求解器返回的时长与公式值不一致（浮点容差等原因），
        # 必须以公式值修正后才允许输出，并同步重算离港时间，
        # 保证 end_time - start_time == service 恒成立
        if has_cargo[i]:
            formula_svc = round(cargos[i] / (cranes_used * rates[i]), 2)
            svc_out = round(float(pulp.value(svc_expr[i])), 2)
            if abs(svc_out - formula_svc) > 1e-9:
                corrections.append(f"{names[i]}: {svc_out}h → {formula_svc}h")
                svc_out = formula_svc
        else:
            svc_out = services[i]

        # 离港时间 = 开工时间 + 校验/修正后的作业时长（保留两位小数）
        end_time = round(start_time + svc_out, 2)
        weighted_total += weights[i] * (end_time - arrivals[i])

        # 岸线位置（连续变量，保留两位小数）
        start_pos = round(float(pulp.value(p[i])), 2)
        end_pos = round(start_pos + lengths[i], 2)

        schedule.append({
            "name": names[i],
            "arrival": arrivals[i],
            "service": svc_out,
            "length": lengths[i],
            "berth": berth_assigned,
            "start_time": start_time,
            "end_time": end_time,
            "wait_time": wait,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "cranes": cranes_used
        })

    # ====== 岸桥使用峰值（模块二） ======
    # 按事件扫描精确计算：船 i 在 [start_time, end_time) 占用 cranes 台岸桥；
    # 同一时刻先处理离港（释放）再处理开工，对应作业区间的左闭右开定义
    crane_peak = 0
    events = []
    for i in range(n):
        if has_cargo[i]:
            events.append((schedule[i]["start_time"], 1,
                           schedule[i]["cranes"]))
            events.append((schedule[i]["end_time"], 0,
                           schedule[i]["cranes"]))
    events.sort(key=lambda e: (e[0], e[1]))  # 0=离港 优先于 1=开工
    cur = 0
    for _, typ, nn in events:
        if typ == 1:
            cur += nn
            crane_peak = max(crane_peak, cur)
        else:
            cur -= nn

    # ====== 求解完成后在函数内部生成甘特图 =====
    # 保存 berth_gantt.png；gantt_image 仅返回本地路径提示文字（不返回 Base64），
    # 绘图异常不影响调度结果返回
    gantt_image = ""
    # 强制校验若发生过时长修正，在 message 中说明
    message = ("作业时长已按装卸量公式强制校验修正: " + "; ".join(corrections)
               if corrections else "")
    try:
        gantt_path, _ = plot_gantt(schedule, num_berths)
        gantt_image = f"甘特图已生成，保存在本地：{gantt_path}"
    except Exception as exc:
        message = (message + "；" if message else "") + f"甘特图生成失败: {exc}"

    return {
        "status": "success",
        "schedule": schedule,
        "total_waiting": total_wait,
        "weighted_total": weighted_total,
        "crane_peak": crane_peak,
        "gantt_image": gantt_image,
        "message": message
    }


@app.route("/optimize", methods=["POST"])
def optimize():
    """
    HTTP API：接收 JSON 输入，返回泊位调度优化结果。
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    ships = data.get("ships", [])
    berths = data.get("berths", 1)

    if not ships:
        return jsonify({"error": "At least one ship is required"}), 400

    # 新增可选字段：JSON 中未提供时默认不启用对应约束。
    # 求解函数内部已完成甘特图绘制（gantt_image 为本地路径提示文字），
    # 直接返回完整 JSON 骨架：
    # {status, schedule, total_waiting, weighted_total, crane_peak,
    #  gantt_image, message}
    result = optimize_berth_scheduling(
        ships,
        berths,
        tide_windows=data.get("tide_windows"),
        dependencies=data.get("dependencies"),
        priority_weights=data.get("priority_weights"),
        berth_restrictions=data.get("berth_restrictions"),
        min_gap=data.get("min_gap", 0.0),
        custom_constraints=data.get("custom_constraints"),
        objective=data.get("objective", "weighted_time"),
        quay_length=data.get("quay_length", 1000),
        total_cranes=data.get("total_cranes"),
        crane_step=data.get("crane_step", 0.5),
    )
    return jsonify(result)


def _print_result(result):
    """打印调度结果（本地测试用）。"""
    if result.get("status") == "infeasible":
        print(f"  模型无解 (infeasible): {result.get('message', '')}")
        return
    for s in result["schedule"]:
        cranes = s.get("cranes")
        print(f"  船舶 {s['name']}: 泊位 {s['berth'] + 1}, "
              f"到达 {s['arrival']}, 开始 {s['start_time']}, "
              f"结束 {s['end_time']}, 等待 {s['wait_time']}, "
              f"位置 [{s.get('start_pos', '-')}~{s.get('end_pos', '-')}米]"
              + (f", 岸桥 {cranes}" if cranes is not None else ""))
    print(f"  总等待时间 total_waiting: {result['total_waiting']}")
    print(f"  加权总在港时间 weighted_total: {result['weighted_total']}")
    print(f"  岸桥使用峰值 crane_peak: {result.get('crane_peak', '-')}")
    print(f"  gantt_image 长度: {len(result.get('gantt_image', ''))} 字符")
    if result.get("message"):
        print(f"  message: {result['message']}")


if __name__ == "__main__":
    # ====== 本地测试 1：基础功能（无额外约束，行为与原版一致） ======
    print("=" * 50)
    print("本地测试 1：3 艘船，2 个泊位（基础功能）")
    print("=" * 50)

    test_ships = [
        {"name": "A", "arrival": 1, "service": 2},
        {"name": "B", "arrival": 2, "service": 1},
        {"name": "C", "arrival": 3, "service": 3}
    ]
    test_berths = 2

    result = optimize_berth_scheduling(test_ships, test_berths)
    _print_result(result)
    print(f"  甘特图已保存: {DEFAULT_GANTT_PATH}")

    # ====== 本地测试 2：包含全部新增字段的完整 JSON（走 /optimize 接口） ======
    print("\n" + "=" * 50)
    print("本地测试 2：完整 JSON（潮汐/依赖/权重/靠泊限制/间隔/硬约束/目标/连续岸线）")
    print("=" * 50)

    full_payload = {
        "ships": [
            {"name": "A", "arrival": 0, "service": 3, "length": 100},
            {"name": "B", "arrival": 1, "service": 2, "length": 80},
            {"name": "C", "arrival": 2, "service": 2, "length": 90},
            {"name": "D", "arrival": 3, "service": 2, "length": 100},
            {"name": "E", "arrival": 4, "service": 3, "length": 120},
            {"name": "F", "arrival": 5, "service": 2, "length": 80},
            {"name": "G", "arrival": 6, "service": 2, "length": 90}
        ],
        "berths": 3,
        # 连续岸线总长（米）：时间上重叠的船在岸线上空间不得重叠
        "quay_length": 500,
        # 岸桥数量：当前模型按固定作业时长求解，此字段暂不参与计算，仅作输入示例
        "berth_cranes": [3, 2, 2],
        # 泊位 0 不可作业区间 [9,11] 和 [15,17]，泊位 1 无，泊位 2 为 [12,14]
        "tide_windows": [[[9, 11], [15, 17]], [], [[12, 14]]],
        "priority_weights": {"A": 3, "B": 2, "C": 1, "D": 2, "E": 1, "F": 0.5, "G": 1},
        # C 需等 A 离港后才能开始；E 需等 B、D 均离港后才能开始
        "dependencies": {"C": ["A"], "E": ["B", "D"]},
        # A 只能停泊位 0/1，F 只能停泊位 1/2
        "berth_restrictions": {"A": [0, 1], "F": [1, 2]},
        # 同泊位前后两船间隔至少 1 小时
        "min_gap": 1.0,
        # A 必须在 16.0 之前离港
        "custom_constraints": [{"type": "deadline", "ship": "A", "time": 16.0}],
        # 目标函数：最小化加权总在港时间
        "objective": "weighted_time"
    }

    client = app.test_client()
    result = client.post("/optimize", json=full_payload).get_json()
    _print_result(result)
    print(f"  甘特图已保存: {DEFAULT_GANTT_PATH}")

    # ====== 本地测试 3：无解场景（应返回 {"status": "infeasible"} 而非崩溃） ======
    print("\n" + "=" * 50)
    print("本地测试 3：无解场景（A 最早 3.0 才能离港，却要求 2.0 前离港）")
    print("=" * 50)

    infeasible_payload = dict(full_payload)
    infeasible_payload["custom_constraints"] = [
        {"type": "deadline", "ship": "A", "time": 2.0}
    ]
    result = client.post("/optimize", json=infeasible_payload).get_json()
    print(f"  返回结果: {result}")

    # ====== 本地测试 4：装卸量动态时长 + 全局岸桥约束（模块一/二） ======
    # 3 艘船、2 个泊位、3 台岸桥：
    #   A/B 集装箱 180 TEU：n=1/2/3 台岸桥时作业时长 6/3/2 小时
    #   C   散货 1200 吨：  n=1/2/3 台岸桥时作业时长 3/1.5/1 小时
    print("\n" + "=" * 50)
    print("本地测试 4：3 艘船、2 泊位、3 台岸桥（装卸量计算时长 + 岸桥峰值约束）")
    print("=" * 50)

    crane_payload = {
        "ships": [
            {"name": "A", "arrival": 0, "cargo": 180,
             "cargo_type": "container", "max_cranes": 3, "length": 100},
            {"name": "B", "arrival": 1, "cargo": 180,
             "cargo_type": "container", "max_cranes": 3, "length": 80},
            {"name": "C", "arrival": 2, "cargo": 1200,
             "cargo_type": "bulk", "max_cranes": 3, "length": 90}
        ],
        "berths": 2,
        # 港口总岸桥数：任意时刻所有在作业船占用的岸桥总数不得超过该值
        "total_cranes": 3,
        # 时间离散化步长（小时）
        "crane_step": 0.5
    }
    result = client.post("/optimize", json=crane_payload).get_json()
    _print_result(result)
    peak = result.get("crane_peak")
    ok = isinstance(peak, (int, float)) and 0 <= peak <= 3
    print(f"  校验 crane_peak <= 3: {'PASS' if ok else 'FAIL'}")

    # ====== 启动 Flask 服务 ======
    print("\nFlask 服务启动在 http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
