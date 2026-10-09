#!/usr/bin/env python3
"""Render the compact Chinese report from saved, scene-paired statistics."""
import json
from pathlib import Path
from collections import Counter
from analyze import OUT,load_rows,MODES,INITS


def pct(x):return f'{100*x:.1f}%'
def interval(ci):return f'[{100*ci[0]:.1f}, {100*ci[1]:.1f}]'
def cell(g):return f"{g['success']}/{g['n']} ({pct(g['success_rate'])}; {interval(g['ci95'])})"
def matrix(groups):
    return '\n'.join(['| 控制器 | G0 | G− | G＋ |','|---|---:|---:|---:|']+
        ['| '+m+' | '+' | '.join(cell(groups[m+'/'+g]) for g in INITS)+' |' for m in MODES])
def contrast(c):return f"{100*c['difference']:+.1f} pp，95% CI {interval(c['ci95'])} pp"


def main():
    stats=json.loads((OUT/'statistics.json').read_text());config=json.loads((OUT/'config.json').read_text())
    assert config['frozen'] and stats['formal']['all_attempts']['rollouts']==600
    formal=stats['formal'];main=formal['all_attempts'];groups=main['groups'];dev=stats['development']['all_attempts']['groups']
    rows=load_rows();test=[r for r in rows if r['round']=='round3'];cal=json.loads((OUT/'calibration.json').read_text())
    geometry=formal['geometry']['initializations'];contrasts=main['paired_contrasts']
    text=['# OrchardBench-VLA：初始化先验与目标空间信息对照（1009）','',
        '## 1. 研究问题与结论范围','',
        '原站位规划器先用真值选择果实，再按高度相关的 standoff band 寻找底座位置，令 yaw 正对目标，并筛选出生碰撞、抓取与回撤等可达性。因此目标基座坐标接近 `(x(z), 0, z)`：横向定位需求被初始化消除，前后距离变化也被压缩。本实验检验剩余目标空间信息的边际价值，没有训练或评测 VLA。','',
        f"正式结果为 50 个新基础场景、600 条 rollout。C0/G0 为 {groups['C0/G0']['success']}/50；Cz/G0 为 {groups['Cz/G0']['success']}/50；Cxyz0/G0 为 {groups['Cxyz0/G0']['success']}/50；Cxyz/G0 为 {groups['Cxyz/G0']['success']}/50。以下主指标均为任意一颗果实的原 StrictPlacement 成功。",'',
        '主实验测量的是**共享特权阶段反馈条件下，目标空间信息的贡献**。C0 仍有机器人本体反馈、规划目标的抓持／脱果反馈、held15、完成标记和真实速率标量，不能称为完全无感知机器人。Cxyz 是持续真值几何参考，不是真实视觉策略。','',
        '## 2. 实验设计与信息隔离','',
        '| 模式 | reset 获得的目标信息 | 后续接近／抓取 | 抓持后位置 |',
        '|---|---|---|---|',
        '| C0 | 无 | 冻结均值模板 | TCP 加全局固定偏移 |',
        '| Cz | 一次高度 z₀ | 冻结 `(x̂(z₀),0,z₀)` | 同上 |',
        '| Cxyz0 | 一次完整三维坐标 | 固定初始坐标 | 同上 |',
        '| Cxyz | 初始三维坐标 | 每 5 个真实控制步更新 | 同频率更新真值 |','',
        f"C0 模板为 `{cal['nominal_base_xyz']}` m；Cz 映射为 `x̂(z)={cal['height_to_x'][0]:.6f}z+{cal['height_to_x'][1]:.6f}` m，独立 40 场景校准的 x 残差 RMSE 为 {100*cal['residual_x_rmse']:.2f} cm。固定抓持偏移为 TCP 系 `(0,0,0)`，不根据 rollout、seed 或成功情况校准。该近似可能损失持果偏移信息，是本轮低信息实现的明确局限。",'',
        '`TargetInformationProvider` 不接收 env、seed 或 planner metadata。C0 的构造接口不接收测量；Cz 只接收标量高度；Cxyz0 只接收一次初始坐标。后三阶段继续使用同一受限估计，低信息组实时位置更新计数为零。`ControlledExpert` 构造白名单后直接调用生产 `StudentNativeExpert.plan()`，没有复制或另写状态机。测试覆盖所有阶段的命令一致性、非法位置输入拒绝、运输末段和坐标变换等变性。','',
        'StageFeedback 与位置接口分离，仅包含规划目标 held / detached、held15_step、strict_success_step 和 fruit_speed 标量。规划目标身份只在评估侧用于形成共同布尔反馈；状态机收到固定合成 ID。桶位置由当前机器人底座位姿和已知安装几何计算。世界坐标仅用于刚体坐标变换，没有利用底座位置反解规划目标。完整位置、其他果实和碰撞真值只用于独立评估日志。','',
        '统一契约：60 Hz 仿真，30 Hz 控制，action_repeat=2，H=5；reach/grasp/pull/transport 为 0.20/0.04/0.08/0.20 m/s，旋转 0.30 rad/s；pregrasp=0.16 m、retract=0.32 m；阶段预算 12/12/8/45/8 s，总预算 2100 控制步。所有缓存命令只执行一次 `native_command → OrchardVLAEnv.step`，无 reach-conditioned、额外 dwell。benchmark_assist 和 detach_force_scale=1.5 不变。','',
        'G−/G＋在原 stance 的局部 y 方向平移 −/+0.08 m，原 yaw 不变；先选原目标，再在 world 构造前覆盖 stance。没有再次优化、teleport 或改初始关节。每次 reset 断言所有初始果实坐标与场景清单一致、home 关节一致、仿真时钟为零。树和物理使用相同 seed 与配置；桶随机器人整体移动。','',
        'Round 0：3 个已有 Gate A 场景比较原专家和 Cxyz，均为 2/3 严格成功；另有每模式 2 条 smoke。第一次审计发现固定底座焊接约束有约 0.33 mm 顺应位移，适配层错误冻结底座坐标导致微米级运输命令差异；唯一修订改用当前机器人自身底座位姿及桶安装关系。修订后同观测命令误差约 10⁻⁸，成功轨迹相差 0/5 控制步。前后结果及工程异常保留，没有调整控制参数。','',
        'Round 1/2 使用 8300000 起的前 20 个原规划器可行场景，分别运行 80/160 次。校准使用独立 8290000 起的 40 个场景。正式名单为 8400000 起的前 50 个可行场景，配置冻结后运行全部 600 次，不根据正式结果重选或调参。所有新场景与 Gate C 和 V2 采集不重叠。具体全部扫描尝试见 scene_manifest.json。','',
        '## 3. 初始几何分布与有效性','',
        '| 初始化 | x 均值±SD (m) | y 均值±SD (m) | z 均值±SD (m) | 果实–TCP距离均值／范围 (m) | 端点IK可达：预抓／抓取 | 出生有效 |',
        '|---|---|---|---|---|---:|---:|']
    for g in INITS:
        d=geometry[g];mu=d['xyz_mean'];sd=d['xyz_std'];dr=d['target_tcp_distance_range']
        text.append(f"| {g} | {mu[0]:.4f}±{sd[0]:.4f} | {mu[1]:.4f}±{sd[1]:.2g} | {mu[2]:.4f}±{sd[2]:.4f} | {d['target_tcp_distance_mean']:.4f} / [{dr[0]:.4f},{dr[1]:.4f}] | {d['endpoint_ik_reachable']['pregrasp']}/{d['endpoint_ik_reachable']['grasp']} | {d['n']-d['invalid']}/{d['n']} |")
    text += ['',
        '有效性使用原规划器同类 chassis、bucket、wheel 和 home-arm 代理几何检查初始重叠；软细枝接触单独记录。端点 PoseIK 使用独立求解器，求解失败不作为排除依据，也不等于证明全局不可达。该检查不保证运动中的碰撞和路径可行性。全部尝试与共同有效场景均报告；原 planner 预筛不通过的候选也保留在 manifest。', '',
        f"正式阶段共有 {sum(geometry[g]['invalid'] for g in INITS)}/150 个无效初始化；共同有效基础场景 {len(formal['common_valid_seeds'])}/50。配对几何检查 {formal['geometry']['paired_geometry_checks']} 次通过：x/z 不变，目标 y 改变为底座扰动的相反数，yaw 不变。共同有效子集与全部尝试的矩阵{'相同' if len(formal['common_valid_seeds'])==50 else '另见 statistics.json'}。",'',
        '## 4. 主要结果','',
        '单格括号依次为成功率及 Wilson 95% 区间（%）。开发集只用于验证框架，正式集才用于冻结后的比较。','',
        '### 开发集（20 个基础场景）','',matrix(dev),'',
        '### 正式集（50 个基础场景）','',matrix(groups),'',
        f"任意果实与规划果实指标不一致的正式 rollout 数：{sum(r['strict_success_any_fruit']!=r['strict_success_planned_fruit'] for r in test)}。没有因采到非规划目标、释放距离超出 0.15 m、IK 或 clipping 质量门槛而抹掉原严格成功。所有果实均由原 observer 连续观察。",'',
        '### 同果事件漏斗与执行诊断','',
        '| 模式/初始化 | 进入GRASP | held15 | detach链 | 有效实际释放 | stable60/strict | 全部步数中位数 | 成功步数中位数 | IK失败率 | clipping率 | 轨迹p95跟踪误差中位数(m) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in MODES:
        for g in INITS:
            d=groups[m+'/'+g];f=d['chain_funnel'];sm=d['success_steps_median']
            text.append(f"| {m}/{g} | {f['grasp_phase']} | {f['held15']} | {f['detach']} | {f['valid_release']} | {f['stable_bucket']} | {d['steps_median']:.0f} | {sm if sm is not None else '—'} | {pct(d['ik_failure_rate'])} | {pct(d['clipping_rate'])} | {d['tracking_position_p95_median']:.4f} |")
    text += ['', '漏斗采用原 StrictPlacement 同果链；detach 链指该果实已满足 held15 且记录 grasp 后 detach。release 原始事件与 valid_chain 分开保存。IK/clipping 为控制步加权比例；较短失败轨迹不能解释为更高执行效率。各阶段起止控制步、首次抓持／释放／成功步数均在 results.csv。','',
        '| 模式/初始化 | 失败原因（次数） | 释放距离 p50/p95 (m) | 释放速率 p50/p95 (m/s) |','|---|---|---|---|']
    for m in MODES:
        for g in INITS:
            d=groups[m+'/'+g];rd=d['release_diagnostics']
            def pair(key):
                v=rd[key];return '—' if v is None else f"{v['median']:.4f}/{v['p95']:.4f}"
            text.append(f"| {m}/{g} | {', '.join(k+': '+str(v) for k,v in sorted(d['failures'].items())) or '无'} | {pair('fruit_bucket_distance')} | {pair('fruit_speed')} |")
    text += ['', '释放距离和速率在实际 release 事件处测量、独立报告，不附加球形成功区域。逐条事件含果实 ID，非规划果实也使用其自身位置和速率。','',
        '### 按基础场景配对的空间信息增益','',
        '| 高信息−低信息 | G0 | G− | G＋ |','|---|---|---|---|']
    for high,low in [('Cz','C0'),('Cxyz0','Cz'),('Cxyz','Cxyz0'),('Cxyz','C0')]:
        text.append('| '+high+'−'+low+' | '+' | '.join(contrast(contrasts[f'{high}-{low}/{g}']) for g in INITS)+' |')
    text += ['', '| 同一控制器的扰动−G0 | G− | G＋ |','|---|---|---|']
    for m in MODES:text.append('| '+m+' | '+' | '.join(contrast(contrasts[f'{m}/{g}-G0']) for g in ['G-','G+'])+' |')
    text += ['',f"信息增益 `SR(Cxyz)−SR(C0)` 相比 G0 的变化：G− 为 {contrast(contrasts['information_gain/G--G0'])}；G＋为 {contrast(contrasts['information_gain/G+-G0'])}。跨三个初始化先在每个 seed 内平均的信息增益为 {contrast(contrasts['Cxyz-C0/averaged_G_clustered'])}。",'',
        '差异使用 20,000 次固定随机种子的基础场景 bootstrap，三种初始化不被当作独立样本。区间未进行多重比较校正，属于本次成对对照的边际区间；区间覆盖零不代表等效，全零差异导致的退化 bootstrap 区间也不能证明总体等效。','',
        '## 5. 结果归因','',
        '<!-- INTERPRETATION: fill after inspecting final results -->','',
        '## 6. 对后续研究的影响','',
        '<!-- DECISIONS: fill after inspecting final results -->','',
        '## 7. 局限、运行与复现','',
        '只有 50 个独立基础场景，单格成功率在中间区域的 95% 区间宽度约 26 个百分点，不能把小差异解释为等效。reset 可行性本身由特权规划器筛选，因此结论只适用于该条件分布；横向变化只有统一 ±8 cm，不能代表任意站位或一般采摘。','',
        'GPU 物理、接触和并行调度可能产生非确定性；每格每场景仅一次，未估计多次重跑的方差。Round 0 比较证明框架保留执行机制，不声称长程轨迹位级一致。出生几何和端点 IK 检查也不能排除扰动改变运动中碰撞、关节姿态或动态可达性。','',
        'Cxyz 的仿真真值没有真实视觉的遮挡、误差、延迟和目标关联问题；低信息组的共享阶段反馈、真实速率和 benchmark_assist 仍有信息及物理优势。Cxyz0 抓持后的固定 TCP 偏移不能代表所有可能的无在线定位预测器。本轮也没有匹配训练的无图像模型，不能从几何控制器结果直接推断 VLA 是否使用图像。','',
        '未执行 Round 4：前三轮已能回答本轮目标空间信息与横向初始化的主要判别问题，没有根据正式集结果继续选择控制器或调参。若以后研究定位刷新频率，应使用新探索场景，并明确区分探索性与确认性证据。','',
        f"运行配置冻结记录见 config.json。原生产基线提交为 `{config['orchard_baseline_commit']}`，XR-0 observer 来源为 `{config['xr0_commit']}`。本轮不修改生产专家、native action/controller、物理、正式采集脚本、gate_status 或 V2 数据。已有 2000 条数据采集在本地开始检查时已完成，本轮未操作其进程与配置。",'',
        '复现入口见 `experiments/initialization_prior_1009/README.md`。核心文件为 controller.py、scenes.py、run.py、analyze.py、audit.py、report.py；CPU 权限测试为 test_information.py。results.csv 包含所有完成的轮次及修订前已完成轨迹；4 条修订前审计中断记录在 engineering_errors.json，原始未完成日志仍留在本地。场景、校准、冻结配置、统计、验证与修订记录随报告保存；压缩逐步日志及运行缓存不提交。','']
    (OUT/'final_report.md').write_text('\n'.join(text))
    print(OUT/'final_report.md')
if __name__=='__main__':main()
