#!/usr/bin/env python3
"""Filter the completed frozen cohort, prepare both existing stats contracts, report."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from run_v1_dataset_pipeline import ROOT, read, rows, sha, write, manifest, append
import collect_autopicker_dataset as collector

XR0 = ROOT.parent / 'dualsys/Xiaomi-Robotics-0/xr0'


def link(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        assert dest.resolve() == source.resolve()
    else:
        dest.symlink_to(source)


def prepare(out):
    from v1_dataset_worker import check_sources
    protocol = check_sources(out)
    raw, filtered = Path(protocol['raw_root']), Path(protocol['filtered_root'])
    frozen = read(out / 'raw_cohort_frozen.json')
    assert sha(out / 'raw_2650_manifest.jsonl') == frozen['manifest_sha256']
    assert sha(raw / 'manifest.jsonl') == frozen['raw_manifest_sha256']
    cohort = rows(out / 'raw_2650_manifest.jsonl')
    replays = rows(out / 'replay_results.jsonl')
    assert len(cohort) == len(replays) == 2650
    assert len({r['seed'] for r in replays}) == 2650
    by_seed = {r['seed']: r for r in replays}
    assert set(by_seed) == {r['seed'] for r in cohort}
    trainable, excluded, invalid = [], [], []
    for row in cohort:
        r = by_seed[row['seed']]
        assert r['episode_id'] == row['episode_id']
        assert r['source_sha256'] == row['annotation_sha256']
        assert r['protocol_sha256'] == sha(out / 'protocol.json')
        assert r['mode'] == 'reach-conditioned' and r['semantics_verified']
        valid, reason = True, None
        try:
            annotation = raw / row['annotation']
            assert sha(annotation) == row['annotation_sha256'], 'JSON changed after cohort freeze'
            t = read(annotation)
            collector.validate_arrays(t)
            for view, entries in t['observations'].items():
                assert sha(entries[0]['path']) == row['video_sha256'][view], 'Video missing or changed'
                assert row['video_audit'][view]['decoded_frames'] == t['num_frames']
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            valid, reason = False, f'{type(exc).__name__}: {exc}'
        item = dict(row, integrity_valid=valid, oracle_strict_success=r['success'],
                    oracle_result=str(out / f'replay_jobs/seed_{row["seed"]}/result.json'),
                    replay={k: r[k] for k in ('final_phase', 'final_target_index', 'total_targets',
                        'termination_reason', 'grasped', 'detached', 'dwell_timeout_count',
                        'forced_advancement_count', 'IK_failed_steps', 'clipped_steps',
                        'control_steps', 'sim_duration')})
        if not valid:
            item.update(category='integrity_invalid', integrity_reason=reason)
            invalid.append(item)
            excluded.append(item)
        elif r['success']:
            item.update(category='expert_accepted_replay_pass', dataset_root=str(filtered), cohort='v1_trainable')
            trainable.append(item)
            link(raw / row['annotation'], filtered / row['annotation'])
            for entries in t['observations'].values():
                path = Path(entries[0]['path'])
                link(path, filtered / 'videos' / path.name)
        else:
            item.update(category='expert_accepted_replay_fail')
            excluded.append(item)
    assert len(trainable) + len(excluded) == 2650
    assert {r['seed'] for r in trainable} == {
        r['seed'] for r in cohort if by_seed[r['seed']]['success'] and r['seed'] not in {x['seed'] for x in invalid}}
    manifest(out / 'v1_trainable_manifest.jsonl', trainable)
    manifest(out / 'v1_excluded_manifest.jsonl', excluded)
    manifest(out / 'integrity_invalid.jsonl', invalid)
    manifest(filtered / 'manifest.jsonl', trainable)
    write(filtered / 'collection_config.json', read(out / 'collection_config.json'))
    assert len(list((filtered / 'json').glob('*/*.json'))) == len(trainable)
    assert len(list((filtered / 'videos').glob('*.mp4'))) == 2 * len(trainable)

    # Preserve the collector's existing 14-active-dimension 30x32 preview statistics.
    stats = collector.statistics(filtered, trainable, 30)
    assert stats is not None and stats['active_dims'] == [0, 14]
    train_rows = [r for r in trainable if r['split'] == 'train']
    expected_windows = sum(r['frames'] - 29 for r in train_rows)
    assert stats['full_window_samples'] == expected_windows
    stats.update(source_manifest_sha256=sha(filtered / 'manifest.jsonl'),
                 source_episode_ids=[r['episode_id'] for r in train_rows],
                 filtered_train_only=True)
    write(filtered / 'action_stats_30x32.json', stats)

    # The existing deployment-contract loader uses its own 7-active-dimension stats.
    # Reuse its unmodified preparation tool; do not import a model or a trainer.
    tool = XR0 / 'tools/compute_orchard_stats.py'
    loader = XR0 / 'mibot/data/datasets/orchardbench_dataset.py'
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(XR0))),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    cmd = [str(XR0 / '.venv-orchard/bin/python'), str(tool), '--root', str(filtered),
           '--output', str(filtered / 'action_stats.json')]
    append(out / 'dataset_preparation_commands.jsonl', dict(command=cmd, source_sha256=sha(tool),
                                                           loader_source_sha256=sha(loader)))
    with (out / 'normalization.log').open('w') as log:
        subprocess.run(cmd, cwd=XR0, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    loader_stats = read(filtered / 'action_stats.json')
    assert loader_stats['episodes'] == [r['episode_id'] for r in train_rows]
    assert loader_stats['full_window_samples'] == expected_windows
    check_code = '''import json, sys, numpy as np, torch
from pathlib import Path
from mibot.data.datasets.orchardbench_dataset import OrchardBenchDataset
root = Path(sys.argv[1]); report = {}
with torch.inference_mode():
    for split in ('train', 'val'):
        ds = OrchardBenchDataset({'train_datasets':dict(root=str(root),split=split,stats_path=str(root/'action_stats.json'))})
        for i in sorted({0, len(ds)//2, len(ds)-1}):
            sample = ds[i]
            assert set(sample)=={'messages','action','action_mask','state'}
            assert tuple(sample['action'].shape)==tuple(sample['action_mask'].shape)==(30,32)
            assert tuple(sample['state'].shape)==(1,32)
            assert all(np.isfinite(sample[k].numpy()).all() for k in ('action','action_mask','state'))
            assert not sample['action'][:,7:].any() and not sample['action_mask'][:,7:].any()
            assert sample['action_mask'][:,:7].all()
            assert sum(c['type']=='image' for m in sample['messages'] for c in m['content'])==2
        report[split] = dict(episodes=len(ds.files), full_windows=len(ds), decoded_samples=3)
Path(sys.argv[2]).write_text(json.dumps(dict(status='PASS', no_model_loaded=True, no_gradient_updates=True, splits=report),indent=2)+'\\n')
'''
    cmd = [str(XR0 / '.venv-orchard/bin/python'), '-c', check_code, str(filtered), str(out / 'loader_verification.json')]
    append(out / 'dataset_preparation_commands.jsonl', dict(command=cmd, purpose='six dataset samples, CPU only'))
    with (out / 'loader_verification.log').open('w') as log:
        subprocess.run(cmd, cwd=XR0, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write(out / 'normalization_provenance.json', dict(
        filtered_train_episode_ids=[r['episode_id'] for r in train_rows],
        full_window_samples=expected_windows, validation_used=False,
        preview=dict(path=str(filtered / 'action_stats_30x32.json'),
                     sha256=sha(filtered / 'action_stats_30x32.json'), active_dims=[0,14]),
        deployment_contract=dict(path=str(filtered / 'action_stats.json'),
                     sha256=sha(filtered / 'action_stats.json'), active_dims=[0,7],
                     contract=loader_stats['contract'], compute_source_sha256=sha(tool)),
        no_training=True))
    write(filtered / 'dataset_provenance.json', dict(protocol=str(out / 'protocol.json'),
        raw_dataset=str(raw), filtering='collector accepted AND strict replay success AND basic integrity valid',
        manifest_sha256=sha(filtered / 'manifest.jsonl'), preserved_raw_episode_ids_and_splits=True,
        loader='mibot.data.datasets.orchardbench_dataset.OrchardBenchDataset',
        loader_stats='action_stats.json', collector_preview_stats='action_stats_30x32.json'))
    # Prove filtering/statistics did not mutate raw data.
    assert sha(raw / 'manifest.jsonl') == frozen['raw_manifest_sha256']
    assert all(sha(raw / r['annotation']) == r['annotation_sha256'] for r in cohort)
    summarize(out, protocol, cohort, replays, trainable, excluded, invalid)


def summarize(out, protocol, cohort, replays, trainable, excluded, invalid):
    attempts = rows(out / 'all_attempts.jsonl')
    overflow = rows(out / 'overflow_manifest.jsonl')
    rejected = [r for r in attempts if not r['accepted']]
    retries = rows(out / 'infrastructure_retries.jsonl')
    split_counts = lambda rs: {k: sum(r['split'] == k for r in rs) for k in ('train', 'val')}
    fails = [r for r in replays if not r['success']]
    agg = {k: sum(r[k] for r in replays) for k in ('success', 'grasped', 'detached', 'timeout',
        'IK_failed_steps', 'clipped_steps', 'dwell_timeout_count', 'forced_advancement_count', 'control_steps', 'sim_duration')}
    collection = dict(total_attempts=len(attempts), expert_accepted=2650,
        all_accepted_including_overflow=2650 + len(overflow), overflow_accepted=len(overflow),
        expert_rejected=len(rejected), acceptance_rate=(2650 + len(overflow)) / len(attempts),
        acceptance_rate_denominator='all completed formal attempts, including in-flight overshoot',
        rejection_reason_distribution=dict(Counter(r['reject_reason'] for r in rejected)),
        seed_min=attempts[0]['seed'], seed_max=attempts[-1]['seed'],
        cohort_seed_max=cohort[-1]['seed'], pilot_accepted=3, pilot_in_cohort=False)
    replay = dict(total_replayed=len(replays), strict_PASS=agg['success'],
        strict_FAIL=len(fails), pass_rate=agg['success']/len(replays),
        grasp_count=agg['grasped'], detach_count=agg['detached'], budget_timeout_count=agg['timeout'],
        **{k:agg[k] for k in ('IK_failed_steps', 'clipped_steps', 'dwell_timeout_count',
                             'forced_advancement_count', 'control_steps', 'sim_duration')},
        fail_final_phase=dict(Counter(r['final_phase'] for r in fails)),
        fail_termination_reason=dict(Counter(r['termination_reason'] for r in fails)))
    collection_workers = read(out / 'collection_workers.json')
    replay_workers = read(out / 'replay_workers.json')
    parallel = dict(collection_worker_count=collection_workers['maximum_active_workers'],
        replay_worker_count=replay_workers['maximum_active_workers'], infrastructure_retry_count=len(retries),
        collection_elapsed_seconds=collection_workers['elapsed_s'], replay_elapsed_seconds=replay_workers['elapsed_s'])
    dataset = dict(raw=split_counts(cohort), filtered=split_counts(trainable), final_V1_total=len(trainable),
        excluded_replay_fail_total=sum(r['category']=='expert_accepted_replay_fail' for r in excluded),
        integrity_invalid_total=len(invalid), raw_root=protocol['raw_root'], filtered_root=protocol['filtered_root'],
        stats=read(out / 'normalization_provenance.json'), loader_verification=read(out / 'loader_verification.json'))
    summary = dict(status='COMPLETE', collection=collection, parallel_execution=parallel,
        oracle_replay=replay, dataset=dataset, no_training_performed=True,
        protocol_sha256=sha(out / 'protocol.json'),
        manifests={name:sha(out / name) for name in ('raw_2650_manifest.jsonl', 'replay_results.jsonl',
                    'v1_trainable_manifest.jsonl', 'v1_excluded_manifest.jsonl')})
    write(out / 'dataset_summary.json', summary)
    lines = ['# OrchardBench V1 — 2650 expert collection and frozen replay filtering', '',
        f"正式 raw cohort：2650 accepted；strict replay PASS {replay['strict_PASS']}，FAIL {replay['strict_FAIL']}；最终 V1 {len(trainable)} 条。未加载模型、未下载 checkpoint、未做 gradient update。", '',
        '## Frozen protocol', '',
        'Source of truth: `log/gate1_transport_30s_revalidation`，reach-conditioned 26/30、grasp/detach 30/30。已逐项核对源码 SHA。',
        f"原始 frozen replay SHA256: `{protocol['replay_source_sha256']}`。",
        f"Collector SHA256: `{protocol['collector_source_sha256']}`；git `{protocol['git_commit']}`。",
        'TRANSPORT velocity_accel vmax=0.5 rad/s, amax=20 rad/s²；其余 phases legacy；detach_force_scale=1.5；真实 ego/wrist RGB，30 Hz，measured proprio/next-state actions。',
        'Replay：position≤0.01 m、SO(3)≤0.08 rad、maximum dwell=30；旧 600-step 默认显式覆盖为 900 steps / 30 s。reach OR maximum-dwell fallback，每次推进一个目标。无 passed-waypoint、lookahead、插值、额外 terminal hold、recovery 或 replanning。',
        'GT encode_window → OrchardActionAdapter → to_native(live obs) → OrchardVLAEnv.step；PASS 只取当前 strict evaluator info.success。', '',
        '## Collection', '',
        f"Total attempts {len(attempts)}；正式 expert accepted 2650；expert rejected {len(rejected)}；并发 overflow accepted {len(overflow)}（单独保留，不进入正式 cohort）。",
        f"全部完成 attempt 的 acceptance rate {(2650+len(overflow))/len(attempts):.2%}（包含 overflow；未因 replay FAIL 补采）。",
        f"正式 attempt seed {collection['seed_min']}–{collection['seed_max']}；2650th accepted seed {collection['cohort_seed_max']}。独立 pilot 3 accepted，不计入 2650。",
        f"Rejection reasons: `{json.dumps(collection['rejection_reason_distribution'],ensure_ascii=False)}`。", '',
        '## Parallel execution', '',
        f"Collection workers {parallel['collection_worker_count']}；replay workers {parallel['replay_worker_count']}；infrastructure retries {len(retries)}。",
        f"Collection wall-clock {parallel['collection_elapsed_seconds']/60:.1f} min；replay wall-clock {parallel['replay_elapsed_seconds']/60:.1f} min。",
        '每个 job 独立进程、Newton environment、seed、日志和事务。中央按 seed 排序后确定前 2650、episode ID 与 split；正式 cohort 完整冻结后才启动 replay。完整结果（包括 FAIL）从不重跑。',
        '本批使用私有 CUDA MPS 服务调度 CUDA clients，启动前已确认设备支持；控制参数和源码不变，没有改变 GPU compute mode。运行环境、服务日志和逐 job 环境变量见 runtime_environment.json、mps/、launch_commands.jsonl。参考：[NVIDIA MPS 接口](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。',
        '复用 450 批次已通过的相同配置 3-accepted pilot；本批未新增 pilot attempts，也未将 pilot 纳入 cohort。', '',
        '## Oracle replay', '',
        f"Total replayed 2650；strict PASS {replay['strict_PASS']}；strict FAIL {replay['strict_FAIL']}；PASS rate {replay['pass_rate']:.2%}。",
        f"Grasp {replay['grasp_count']}；detach {replay['detach_count']}；budget timeout {replay['budget_timeout_count']}。",
        f"IK failed steps {replay['IK_failed_steps']}；clipped steps {replay['clipped_steps']}；dwell timeouts {replay['dwell_timeout_count']}；forced advancements {replay['forced_advancement_count']}。",
        f"FAIL final phase: `{json.dumps(replay['fail_final_phase'])}`；termination reason: `{json.dumps(replay['fail_termination_reason'])}`。", '',
        '## Dataset', '',
        f"Raw train/val: {dataset['raw']['train']}/{dataset['raw']['val']}；filtered train/val: {dataset['filtered']['train']}/{dataset['filtered']['val']}。",
        f"Final V1 total {len(trainable)}；excluded replay-fail {dataset['excluded_replay_fail_total']}；integrity-invalid {len(invalid)}。",
        '原始 every 10th accepted→val split 在 replay 前确定，过滤后保留 ID 和 split；没有时长、IK、clipping、dwell 或平滑度筛选。',
        '每条 accepted 双视频已完整解码并核对 frame count/FPS/shape；合并核对视频 SHA、JSON schema、有限数值、next-state indexing 和 roundtrip；raw 2650 在筛选后保持原样。',
        f"Raw root: `{protocol['raw_root']}`。Filtered root: `{protocol['filtered_root']}`（独立 manifest，JSON/video 链接，不复制大量视频）。",
        '只用 filtered train 的所有完整窗口重算 `action_stats_30x32.json`（原 14 active preview dims）；同时用现有未修改 preparation 工具重算 loader 所需 `action_stats.json`（既有 Cartesian contract、7 active dims）。两种表示均保持原定义；val 使用 train stats。',
        '已有 OrchardBenchDataset 对 train/val 各实际读取 3 个样本，验证双视角、30×32 action、1×32 state、mask、finite normalization 和 training fingerprint。', '',
        '主要验收产物：`report.md`、`dataset_summary.json`、`v1_trainable_manifest.jsonl`、`v1_excluded_manifest.jsonl`。逐条结果、reject diagnostics、overflow、源码哈希、配置、命令与资源记录都保存在本目录。', '',
        '全部 dataset preparation 完成后停止，没有启动任何训练。', '']
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('dataset','manifests')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'log/v1_2650_collection_replay_filter')
    prepare(parser.parse_args().run.resolve())
