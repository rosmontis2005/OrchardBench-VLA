#!/usr/bin/env python3
"""Report support: baseline repeatability and standalone plots for fresh pairs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_arm_retime_rounds import OUT, load, per_run, identity
from run_arm_retime_pilot import write
from run_arm_retime_rounds import prepare


def repeatability():
    protocol=prepare();rows=[]
    for seed in protocol['smoke']:
        s=seed['seed'];base=load(OUT/f'round0/runs/legacy/{s}/run.json')
        for path in sorted(OUT.glob(f'round*/runs/legacy/{s}/run.json')):
            if path==Path(base['path']):continue
            r=load(path);bp=base['trajectory']['proprios'];rp=r['trajectory']['proprios'];n=min(base['trajectory']['num_frames'],r['trajectory']['num_frames'])
            bt=base['result']['state_trace'];rt=r['result']['state_trace'];same_chain=[x['state'] for x in bt]==[x['state'] for x in rt]
            br=np.array(bp['ee_rotm'][:n]).reshape(-1,3,3);rr=np.array(rp['ee_rotm'][:n]).reshape(-1,3,3)
            rows.append(dict(seed=s,case_type=seed['case_type'],repeat=path.parts[-5],
                baseline_success=base['result']['accepted'],repeat_success=r['result']['accepted'],pair_setup=identity(base,r),
                phase_chain_equal=same_chain,phase_frame_differences=[y['frame']-x['frame'] for x,y in zip(bt,rt)] if same_chain else None,
                same_timestamp_tcp_max_coordinate_error_m=float(np.abs(np.array(bp['ee_pos'][:n])-np.array(rp['ee_pos'][:n])).max()),
                same_timestamp_joint_max_error_rad=float(np.abs(np.array(bp['arm_joint'][:n])-np.array(rp['arm_joint'][:n])).max()),
                same_timestamp_rotation_max_error_rad=float(np.linalg.norm(Rotation.from_matrix(br.transpose(0,2,1)@rr).as_rotvec(),axis=1).max()),
                repeat_failure_reason=r['result'].get('reject_reason'),duration_difference_s=r['result']['sim_duration_s']-base['result']['sim_duration_s']))
    write(OUT/'legacy_repeatability.json',dict(reference='Fresh round0; no canonical or numerical gate; no time warping.',pairs=rows))
    return rows


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    docs=[json.loads(p.read_text()) for p in sorted(OUT.glob('round[123]_summary.json'))]
    if not docs:return
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    metrics=[('tcp_speed','m/s'),('linear_acceleration','m/s²'),('joint_speed','rad/s')]
    for ax,(metric,unit) in zip(axes,metrics):
        x=np.arange(len(docs));b=[d['phase_statistics']['all']['legacy']['TRANSPORT']['metrics'][metric]['P95'] for d in docs];c=[d['phase_statistics']['all']['candidate']['TRANSPORT']['metrics'][metric]['P95'] for d in docs]
        ax.bar(x-.18,b,.36,label='Fresh legacy');ax.bar(x+.18,c,.36,label='Candidate');ax.set_xticks(x,[d['spec']['profile'].replace('transport_','T ').replace('va1.5_a20','v1.5+a20')+'\nN='+str(d['phase_statistics']['all']['candidate']['TRANSPORT']['episodes_with_phase'])+'/8' for d in docs]);ax.set_ylabel(unit);ax.set_title('TRANSPORT '+metric+' P95');ax.legend()
    fig.suptitle('N = candidates reaching TRANSPORT / attempts; T = TRANSPORT-only',fontsize=10)
    fig.tight_layout();fig.savefig(OUT/'transport_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['tcp_speed','linear_acceleration']):
        for d in docs:
            for tag in ['legacy','candidate']:
                data=d['transition_statistics']['all'][tag]['aligned'][metric];xs=sorted(int(k) for k in data);ys=[data[str(k)]['P50'] for k in xs]
                ax.plot(np.array(xs)/30,ys,label=f"R{d['spec']['round']} {tag if tag=='legacy' else d['spec']['profile']}",linestyle='--' if tag=='legacy' else '-')
        ax.axvline(0,color='black',alpha=.3);ax.set_xlabel('s from first fully TRANSPORT interval');ax.set_title(metric+' (median across seeds)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'pull_transport_transition.png',dpi=180);plt.close(fig)



def fmt(x):
    if x is None:return '—'
    return f'{x:.4g}' if isinstance(x,(float,np.floating)) else str(x)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])


def make_report():
    protocol=prepare()
    docs=[json.loads((OUT/'round0_legacy_summary.json').read_text())]+[json.loads(p.read_text()) for p in sorted(OUT.glob('round[123]_summary.json'))]
    decisions=json.loads((OUT/'decisions.json').read_text());final=decisions['final'];best=next(d for d in docs if d['spec']['profile']==final['best_profile'])
    repeats=repeatability();figures()
    def met(d,tag,phase,metric,key='P95',kind='all'):
        return d['phase_statistics'][kind][tag][phase]['metrics'][metric][key]
    def med(d,tag,key):return float(np.median([r[tag+'_'+key] for r in d['runs'] if r.get(tag+'_'+key) is not None]))
    def ratio(d,key):return d['paired_ratios']['all'][key+'_candidate_vs_legacy_ratio']['P50']
    text=['# Arm retime iterative repair test','',f"**Decision: {final['decision']}**",'',
        '## 1. Objective','',
        '复用现有 arm limiter 做有限、逐轮的工程测试；Round 1 系统性接触失败后增加可选 TRANSPORT scope：在保持摘果任务成功的前提下，降低 TRANSPORT 动态与 student action envelope 的差距。没有采集完整数据集、训练、修改 action/observation semantics、tree physics、grasp/hold mechanics、成功定义或 fixed-base 工作位策略。','',
        '## 2. Existing diagnosis','',
        '已读取原 speed_audit.md、summary.json、episode_phase_statistics.csv、episode_statistics.csv，以及上一轮 arm_retime_pilot.md、failure_summary.json、final_summary.txt 和 runner/limiter/analysis 源码。300 accepted episodes 的 TRANSPORT TCP P50/P95/max 为 1.414/2.719/4.249 m/s；高速是系统性问题。旧 pilot 只运行了三个 legacy，因 1000074 的历史逐点比较失败而停止，没有候选结果。','',
        '`AutoPicker._set_arm` 保存 IK q_goal；`_slew_arm` 每个 60 Hz physics frame 将 q_cmd 向 goal 移动，legacy 每轴最多 0.045 rad，即 2.7 rad/s。TRANSPORT 每 15 帧从固定底座计算桶上方目标并求 IK，首次新目标在 phase 切换后一帧发出；IK 以当前 q_cmd 为 seed。因此限速也可能改变后续 IK 解与接触轨迹。现有 opt-in limiter 替换七个 arm joint command，未限制 IK goal 本身、未改动力学状态或夹爪。velocity/acceleration 状态跨 phase 保留。','',
        'student `VLAConfig` 保留 translation 每轴 0.02 m/control action、rotation 相对 world XYZ Euler 每轴 0.05 rad/control action、joint slew 0.045 rad/physics frame；physics 60 Hz、repeat=2，因此 control 30 Hz。expert 的 joint command 限速并不保证 TCP translation/rotation 完全落入此 envelope，也不保证 measured joint velocity 不超过 command limit。','',
        '## 3. Experimental protocol','',
        '复用 `run_arm_retime_pilot.one` / `collect_episode`、原 ArmMotionProfile 和原 motion/aggregate/transition/diagnostics 数值函数。Round 0 对全部八个 seed 建立 fresh legacy；每个候选轮次再次按 seed 顺序运行 fresh legacy → candidate（独立 subprocess）。同 seed、same world config、apple selection、reset planner、60 Hz / 3 substeps、detach_force_scale=1.5；record_rgb=False，但原 visibility acceptance 检查保留。','',
        '配对检查选中果实索引/初始位置/半径、base pose、standoff/azimuth、初始 TCP/arm/gripper、seed 与 detach force multiplier；每轮底层源码 SHA256 在轮前轮后核验。Round 1 后仅在 arm_motion.py/picker.py 增加可选 scope，变更前快照与 source_revision.json 保留；同一轮的 legacy/candidate 使用同一源码。历史 canonical 只作为背景，不参与通过/停止判断。所有失败 retained。','',
        'success 严格使用原 collector accepted：第一次 grasp/detach/placed 完整，无 first-attempt fail、base drift、branch break、wrong target、incidental detach、visibility 或 premature-detach rejection。`DONE` 本身不是 success。记录 20 s overall timeout、PULL timeout，以及 TRANSPORT >420 帧/大 stall 后直接 DROP 的回退迹象。','',
        '运动主表来自 actual 30 Hz proprios：TCP vector finite differences、SO(3) log angular velocity、max abs joint finite-difference velocity；加速度为相邻 velocity 的 raw differences。phase 用 interval 起点，奇数 physics frame 的混合边界归旧 phase；transition 取最后五个完全在边界前、最先五个完全在边界后的 interval，另保留 [-10,+20] 对齐序列。command 60 Hz trace 是 physics step 后、expert.update 后，不能与同帧 measured state 混作因果同时量。Pooled-frame quantiles 与每 seed 配对倍率同时报告，不对失败或慢轨迹静默筛选。','',
        '## 4. Seeds and stress-case definition','',
        table(['Seed','原 selection role','Case','历史 TCP P95 (m/s)','历史 transport path (m)'],[[r['seed'],r['selection_reason'],r['case_type'],r['p95_tcp_speed'],r['transport_path_m']] for r in protocol['smoke']]),'',
        '1000074 在实验前按历史数值敏感性及极端动态标为 stress。1000268 最初为 normal，但 Round 0/1 两次 fresh legacy 在完全相同 setup 下，GRASP→PULL 相差 44 physics frames、DROP 相差 86 frames，总时长 5.10→6.533 s；均成功。仅依据此 legacy-only 证据补标 stress（case_overrides.json），最终 normal=6、stress=2。原始分类仍保留在 protocol.json：Round 1 原 normal 1/7、stress 0/1；新分类 normal 1/6、stress 0/2，系统性失败结论不变。无 seed 删除/替换，也不因 candidate 失败本身改类。','',
        '## 5. Round 0 — fresh legacy','',
        f"任务：normal {docs[0]['cohorts']['normal']['legacy']['success_count']}/6，stress {docs[0]['cohorts']['stress']['legacy']['success_count']}/2；所有八个完整 phase chain，branch break / premature detach 均为 0。",'',
        table(['Phase','TCP P50/P95/max (m/s)','angular P50/P95/max (rad/s)','joint P50/P95/max (rad/s)','TCP acc P95/max (m/s²)'],[[ph,*[' / '.join(fmt(met(docs[0],'legacy',ph,k,q)) for q in ['P50','P95','max']) for k in ['tcp_speed','angular_speed','joint_speed']],' / '.join(fmt(met(docs[0],'legacy',ph,'linear_acceleration',q)) for q in ['P95','max'])] for ph in ['REACH','GRASP','PULL','TRANSPORT','DROP']]),'']
    for d in docs[1:]:
        idx=d['spec']['round'];text += [f"## {6 if idx==1 else 6+idx}. Round {idx} — {d['spec']['profile']}",'',d['spec']['reason'],'',
            table(['Cohort','Fresh legacy success','Candidate success','Candidate failures'],[[kind,f"{d['cohorts'][kind]['legacy']['success_count']}/{d['cohorts'][kind]['legacy']['N']}",f"{d['cohorts'][kind]['candidate']['success_count']}/{d['cohorts'][kind]['candidate']['N']}",json.dumps(d['cohorts'][kind]['candidate']['failure_types'])] for kind in ['normal','stress','all']]),'',
            ('注意：本轮仅 1/8 candidate 到达 TRANSPORT，下方 pooled legacy/candidate 样本覆盖不同；不能将 pooled ratio 解释为全 cohort 改善。失败导致的短 duration 也不是效率收益。' if idx==1 else '下表保留所有尝试的 measured intervals，未筛除失败。'),'',
            table(['TRANSPORT metric','fresh legacy','candidate','pooled ratio'],[[k+' '+q,met(d,'legacy','TRANSPORT',k,q),met(d,'candidate','TRANSPORT',k,q),met(d,'candidate','TRANSPORT',k,q)/met(d,'legacy','TRANSPORT',k,q)] for k in ['tcp_speed','angular_speed','joint_speed','linear_acceleration','joint_acceleration'] for q in ['P50','P95','max']]),'',
            f"TRANSPORT duration median {fmt(med(d,'legacy','TRANSPORT_duration'))} → {fmt(med(d,'candidate','TRANSPORT_duration'))} s；paired duration ratio median {fmt(ratio(d,'TRANSPORT_duration'))}。Episode duration median {fmt(med(d,'legacy','duration'))} → {fmt(med(d,'candidate','duration'))} s；paired ratio median {fmt(ratio(d,'duration'))}。",'',
            table(['Seed','Case','Success L→C','Reason','duration L→C (s)','PULL duration L→C','TRANSPORT duration L→C','TCP P95 ratio','acc P95 ratio'],[[r['seed'],r['case_type'],f"{r['legacy_success']} → {r['candidate_success']}",r['candidate_failure_reason'] or 'none',f"{fmt(r['legacy_duration'])} → {fmt(r['candidate_duration'])}",f"{fmt(r['legacy_PULL_duration'])} → {fmt(r['candidate_PULL_duration'])}",f"{fmt(r['legacy_TRANSPORT_duration'])} → {fmt(r['candidate_TRANSPORT_duration'])}",r.get('TRANSPORT_tcp_speed_P95_candidate_vs_legacy_ratio'),r.get('TRANSPORT_linear_acceleration_P95_candidate_vs_legacy_ratio')] for r in d['runs']]),'']
        dec=decisions.get(f'after_round{idx}')
        if dec and dec.get('confirmation'):text += ['确认结果：'+dec['confirmation'],'']
        if dec:text += [f"## {7 if idx==1 else str(6+idx)+'.1'}. Decision after Round {idx}",'',dec['analysis'],'',f"Next action: {dec['next_action']}",'']
    text += ['## 10. Normal vs stress cases','',
        table(['Profile','Normal success','Stress success','All success','All placed/detached'],[[d['spec']['profile'],*[f"{d['cohorts'][k]['candidate']['success_count']}/{d['cohorts'][k]['candidate']['N']} ({100*d['cohorts'][k]['candidate']['success_rate']:.1f}%)" for k in ['normal','stress','all']],f"{d['cohorts']['all']['candidate']['placed']}/{d['cohorts']['all']['candidate']['detached']}"] for d in docs]),'',
        '每轮 legacy 均独立运行，最终 normal/stress 的 denominator 在各轮分析中统一为 6/2；原分类与补标证据同时保留。以上为有意挑选的 smoke cohort，不是随机总体 success-rate 估计。','',
        'Fresh legacy repeatability（相对 Round 0；同时间比较，phase timing 不同会放大该数值，不是判定 gate）：','',
        table(['Seed','Repeat','Success','phase frame Δ','TCP max coord Δ (m)','joint max Δ (rad)'],[[r['seed'],r['repeat'],r['repeat_success'],r['phase_frame_differences'],r['same_timestamp_tcp_max_coordinate_error_m'],r['same_timestamp_joint_max_error_rad']] for r in repeats]),'',
        '## 11. Failure analysis','',final['failure_analysis'],'',
        table(['Round/profile','branch breaks','premature detach','timeout','transport timeout fallback','possible stall fallback'],[[d['spec']['profile'],sum(r['candidate_branch_break_count'] for r in d['runs']),sum(r['candidate_premature_detach'] is True for r in d['runs']),sum(r['candidate_timeout'] for r in d['runs']),sum(r.get('candidate_transport_timeout_fallback',False) for r in d['runs']),sum(r.get('candidate_transport_stall_fallback_possible',False) for r in d['runs'])] for d in docs]),'',
        'TRANSPORT fallback 检查使用完整 command trace 的 elapsed/stall counter；它是诊断标志，不替换原成功定义。若进入 DROP 的同帧 counter 被 reset，max stall 可少一帧，因此同时保留完整 trace 供复查。','',
        '## 12. Dynamic comparison','','Round 1 仅一个 candidate 到达 TRANSPORT，不能将其 pooled quantile 与八个 legacy 当成无偏改善率；其逐 seed 配对值仅代表该幸存 case。后续完整 cohort 才用于工程推荐。','',
        '下表为每个 seed candidate/fresh-legacy 倍率的中位数。不同于 pooled-frame P95 比值，避免较慢 episode 的更多帧主导唯一结论。','',
        table(['Profile','TCP P95 ratio','TCP acc P95 ratio','joint P95 ratio','PT post TCP ratio','PT post acc ratio','episode duration ratio'],[[d['spec']['profile'],*[ratio(d,k) for k in ['TRANSPORT_tcp_speed_P95','TRANSPORT_linear_acceleration_P95','TRANSPORT_joint_speed_P95','PT_tcp_speed_post','PT_linear_acceleration_post','duration']]] for d in docs[1:]]),'',
        table(['Profile / cohort','TRANSPORT TCP P95 L→C','TRANSPORT acc P95 L→C'],[[d['spec']['profile']+' / '+kind,*[f"{fmt(met(d,'legacy','TRANSPORT',k,kind=kind))} → {fmt(met(d,'candidate','TRANSPORT',k,kind=kind))}" for k in ['tcp_speed','linear_acceleration']]] for d in docs for kind in ['normal','stress']]),'',
        table(['Profile','PT TCP pre→post median (m/s)','PT acc pre→post median (m/s²)','PT command actual acc median max (rad/s²)','PT q_goal jump median (rad norm)','transport >2 m/s frame fraction median'],[[d['spec']['profile'],f"{fmt(med(d,'candidate','PT_tcp_speed_pre'))} → {fmt(med(d,'candidate','PT_tcp_speed_post'))}",f"{fmt(med(d,'candidate','PT_linear_acceleration_pre'))} → {fmt(med(d,'candidate','PT_linear_acceleration_post'))}",med(d,'candidate','PT_command_actual_acceleration_max'),med(d,'candidate','PT_q_goal_jump_max'),med(d,'candidate','TRANSPORT_high_speed_gt_2_fraction')] for d in docs]),'',
        '推荐配置 command limiter 诊断（acceleration exceptions 只在 active scope 内计数；crossing_count 为逐关节累计）：','',
        table(['Seed','crossing count','acc bound exceptions','exceptions without landing','max internal acc in scope','PT actual command acc max'],[[x['seed'],x['crossing_count'],x.get('acceleration_bound_exceptions'),x.get('exceptions_without_landing'),x.get('max_internal_acceleration_in_scope'),next(r['candidate_PT_command_actual_acceleration_max'] for r in best['runs'] if r['seed']==x['seed'])] for x in best['command_watchdog_diagnostics']['candidate']['command_diagnostics']]),'',
        '![TRANSPORT comparison](transport_comparison.png)','', '![PULL to TRANSPORT](pull_transport_transition.png)','',
        '推荐 profile 各 phase 的保留情况与动态（全部尝试，包括失败）：','',
        table(['Phase','N L/C','TCP P95 L→C','angular P95 L→C','joint P95 L→C','duration median L→C'],[[ph,f"{best['phase_statistics']['all']['legacy'][ph]['episodes_with_phase']}/{best['phase_statistics']['all']['candidate'][ph]['episodes_with_phase']}",*[f"{fmt(met(best,'legacy',ph,k))} → {fmt(met(best,'candidate',ph,k))}" for k in ['tcp_speed','angular_speed','joint_speed']],f"{fmt(med(best,'legacy',ph+'_duration'))} → {fmt(med(best,'candidate',ph+'_duration'))}"] for ph in ['REACH','GRASP','PULL','TRANSPORT','DROP']]),'',
        'Student envelope 残余差距：按当前 VLA world-relative Euler clipping 的每轴阈值检查 measured 30 Hz motion，只作对齐诊断，不作新 acceptance gate。','',
        table(['Profile','TRANSPORT translation exceed median','rotation exceed median','either exceed median'],[[d['spec']['profile'],*[med(d,'candidate','TRANSPORT_student_'+k+'_exceed_fraction') for k in ['translation','rotation','either']]] for d in docs]),'',
        '## 13. Recommended profile','',f"**{final['best_profile']}**",'',final['recommendation'],'',
        '## 14. Whether to proceed to 30-seed validation','',f"**{final['decision']}**",'',final['next_action'],'',
        '## 15. Remaining risks','',final['risks'],'',
        'Scope 修复只改变七关节 command retiming 的启用范围；默认 active_phases=None 和无 opt-in profile 的 legacy 行为保持。推荐 profile 仍是实验配置，没有把它悄悄设为全部采集任务默认值。','',
        'Acceleration limiter 的 landing guard 在 goal 落于 stopping distance 内时可能牺牲 acceleration bound；现有实现显式记录 landing/crossing。只有运行过的 profile 才有性能结论。Command derivative 与 measured acceleration 是不同量；30 Hz raw finite differences 不是全部 60 Hz 瞬时峰值。','',
        '## Artifacts and reproduction','',
        '- `protocol.json`、`case_overrides.json`、`seed_geometry.json`、每轮 `spec.json`：固定 seeds、输入源码 hash、profile 与运行前 reason。',
        '- `round0_legacy_summary.json`、实际运行的 `roundN_summary.json`：all/normal/stress phase statistics、transition statistics、配对结果、command/watchdog 诊断。',
        '- `paired_seed_results.csv`：每 seed、每轮的任务、所有 phase P50/P95/max、加速度、duration、transition、limiter saturation、student envelope exceed 和配对倍率。',
        '- `roundN/runs/<profile>/<seed>/run.json`、`stdout.log`：完整 measured trajectory、command trace、reset/acceptance diagnostics，成功失败全量保留。',
        '- `decisions.json`：实际上一轮结果驱动的中间决策与最终决策；`legacy_repeatability.json`：fresh legacy 重复诊断。',
        '- `source_integrity.json`、`limiter_tests.log`、`validation.json`：每轮输入源码一致、允许的 scope 修改另行记录，原六项及新增 scope 测试共七项通过、配对与统计核验。',
        '- 新增实验源码：`scripts/run_arm_retime_rounds.py`、`scripts/analyze_arm_retime_rounds.py`、`scripts/report_arm_retime_rounds.py`。另有最小 limiter 集成修改 `treesim/arm_motion.py`、`treesim/picker.py` 和作用范围回归测试 `scripts/test_arm_motion.py`。physics/student/collector/fixed-base stance 与 phase transitions 未改。','',
        '解释器 `.pixi/envs/default/bin/python`。从仓库根运行 `scripts/run_arm_retime_rounds.py run --round N --profile PROFILE --reason REASON`；每轮 spec 不可覆盖，已完成 runs 可续跑。分析可用 `scripts/analyze_arm_retime_rounds.py log/arm_retime_rounds/roundN`，最终报告用 `scripts/report_arm_retime_rounds.py`。旧 pilot 结果未覆盖。','']
    (OUT/'arm_retime_rounds.md').write_text('\n'.join(text))
    speed=met(best,'candidate','TRANSPORT','tcp_speed')/met(best,'legacy','TRANSPORT','tcp_speed')-1
    acceleration=met(best,'candidate','TRANSPORT','linear_acceleration')/met(best,'legacy','TRANSPORT','linear_acceleration')-1
    summary=f"""ARM RETIME ITERATIVE TEST: {final['decision']}

Best profile: {final['best_profile']}
Normal cases: {best['cohorts']['normal']['candidate']['success_count']}/{best['cohorts']['normal']['candidate']['N']}
Stress cases: {best['cohorts']['stress']['candidate']['success_count']}/{best['cohorts']['stress']['candidate']['N']}
TRANSPORT speed change: pooled TCP P95 {100*speed:+.1f}% ({met(best,'legacy','TRANSPORT','tcp_speed'):.3f} -> {met(best,'candidate','TRANSPORT','tcp_speed'):.3f} m/s)
TRANSPORT acceleration change: pooled TCP acceleration P95 {100*acceleration:+.1f}% ({met(best,'legacy','TRANSPORT','linear_acceleration'):.3f} -> {met(best,'candidate','TRANSPORT','linear_acceleration'):.3f} m/s^2)
Episode duration change: median paired {100*(ratio(best,'duration')-1):+.1f}% ({med(best,'legacy','duration'):.3f} -> {med(best,'candidate','duration'):.3f} s cohort medians)
New failure modes: {final['new_failure_modes']}
Recommended next action: {final['next_action_short']}

Report:
{OUT}/arm_retime_rounds.md
"""
    (OUT/'final_summary.txt').write_text(summary)
    print(summary)


if __name__=='__main__':
    decision_path=OUT/'decisions.json'
    if decision_path.exists() and 'final' in json.loads(decision_path.read_text()):
        make_report()
    else:
        rows=repeatability();figures()
        for r in rows:print(r['seed'],r['repeat'],r['repeat_success'],r['phase_frame_differences'],r['same_timestamp_tcp_max_coordinate_error_m'])
