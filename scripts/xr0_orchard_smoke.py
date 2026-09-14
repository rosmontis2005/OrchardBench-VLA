#!/usr/bin/env python
"""Short XR-0 transport smoke test with an isolated model subprocess.

Parent: official OrchardBench Pixi Python. Child: existing XR-0 Python.
IPC: JSON lines over private pipes, two lossless RGB PNGs, decoded JSON actions.
No public port, Torch dependency in simulator, policy training or quantization.
"""
from pathlib import Path
import sys,argparse,json,os,time,subprocess,select,traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

DEFAULT_MODEL='/home/rosmontis/Projects/dualsys/Xiaomi-Robotics-0/checkpoints/Xiaomi-Robotics-0-Calvin-ABCD_D'
DEFAULT_PYTHON='/home/rosmontis/miniconda3/envs/xr0-mibot/bin/python'


def gpu_memory():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()


def worker(args):
    protocol=sys.stdout
    sys.stdout=sys.stderr  # third-party load diagnostics cannot corrupt JSON IPC
    def reply(data):
        protocol.write(json.dumps(data,allow_nan=False)+'\n');protocol.flush()
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import AutoModel,AutoProcessor
        from treesim.xr0_adapter import xr0_prompt
        t=time.monotonic()
        processor=AutoProcessor.from_pretrained(args.model,trust_remote_code=True,use_fast=False,local_files_only=True)
        model=AutoModel.from_pretrained(args.model,trust_remote_code=True,
            attn_implementation='flash_attention_2',dtype=torch.bfloat16,local_files_only=True).cuda().eval()
        torch.cuda.synchronize()
        reply(dict(status='ready',model=str(args.model),load_s=time.monotonic()-t,
            model_dtype=str(model.dtype),model_device=str(model.device),gpu_memory_mib=gpu_memory(),
            state_dim=model.config.state_dim,config_action_length=model.config.action_length,
            action_mask_shape=list(processor.get_action_mask('calvin_abcd_orig').shape),
            torch_allocated_mib=torch.cuda.memory_allocated()/2**20))
        for line in sys.stdin:
            req=json.loads(line)
            if req.get('stop'):break
            base=Image.open(req['base']).convert('RGB');wrist=Image.open(req['wrist_left']).convert('RGB')
            state=np.asarray(req['state'],dtype=np.float32)
            assert state.shape==(32,) and np.isfinite(state).all()
            inputs=processor(text=[xr0_prompt(req['language'])],images=[base,wrist],videos=None,
                             padding=True,return_tensors='pt').to(model.device)
            inputs['state']=torch.from_numpy(state).to(model.device,model.dtype).view(1,1,-1)
            inputs['action_mask']=processor.get_action_mask(req['task_id']).to(model.device,model.dtype)
            inputs['seed']=int(req['seed'])
            shapes={k:list(v.shape) for k,v in inputs.items() if hasattr(v,'shape')}
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            with torch.inference_mode():
                outputs=model(**inputs)
                decoded=processor.decode_action(outputs.actions,robot_type=req['task_id'])
            torch.cuda.synchronize()
            assert torch.isfinite(outputs.actions).all() and torch.isfinite(decoded).all()
            actions=decoded.float().cpu().numpy()
            reply(dict(status='ok',input_shapes=shapes,raw_shape=list(outputs.actions.shape),
                decoded_shape=list(actions.shape),dtype=str(actions.dtype),finite=True,
                minimum=float(actions.min()),maximum=float(actions.max()),
                gripper_values=actions[0,:,6].tolist(),actions=actions.tolist(),
                inference_s=time.monotonic()-t,gpu_memory_mib=gpu_memory(),
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
    except Exception as exc:
        traceback.print_exc()
        reply(dict(status='error',error=type(exc).__name__,message=str(exc)))
        return 1
    return 0


def recv(proc,timeout=180):
    # Child writes one complete JSON line per reply; buffering disabled in Popen.
    if not select.select([proc.stdout],[],[],timeout)[0]:raise TimeoutError('XR-0 response timeout')
    line=proc.stdout.readline()
    if not line:raise RuntimeError(f'XR-0 child exited {proc.poll()}')
    data=json.loads(line)
    if data.get('status')=='error':raise RuntimeError(f"XR-0: {data['error']}: {data['message']}")
    return data


def parent(args):
    import numpy as np
    from treesim.vla_env import OrchardVLAEnv
    from treesim.xr0_adapter import CalvinActionAdapter,calvin_state,xr0_images
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','cycles':[],'steps':[],'gpu_before_model_mib':gpu_memory()}
    env=OrchardVLAEnv();proc=None
    error=None
    with (out/'xr0_worker.txt').open('w') as logs:
        try:
            proc=subprocess.Popen([args.model_python,'-u',str(Path(__file__).resolve()),'--worker','--model',args.model],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs,text=True,bufsize=1,
                env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'))
            report['model']=recv(proc,args.timeout)
            print('XR0 MODEL',json.dumps(report['model']),flush=True)
            obs,info=env.reset(seed=args.seed)
            report['gpu_after_sim_build_mib']=gpu_memory()
            adapter=CalvinActionAdapter();adapter.reset(obs)
            for cycle in range(args.cycles):
                base,wrist=xr0_images(obs)
                assert not np.array_equal(np.asarray(base),np.asarray(wrist))
                bp=out/f'xr0_base_cycle{cycle}.png';wp=out/f'xr0_wrist_cycle{cycle}.png'
                base.save(bp);wrist.save(wp)
                if cycle==0:
                    base.save(out/'xr0_base_input.png');wrist.save(out/'xr0_wrist_input.png')
                request=dict(task_id='calvin_abcd_orig',state=calvin_state(obs).tolist(),
                    base=str(bp),wrist_left=str(wp),language='Pick an apple.',seed=args.seed+cycle)
                proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
                prediction=recv(proc,args.timeout)
                chunk=np.asarray(prediction.pop('actions'),dtype=np.float32)
                assert chunk.ndim==3 and chunk.shape[0]==1 and chunk.shape[1]>=args.actions_per_cycle and chunk.shape[2]>=7
                assert np.isfinite(chunk).all()
                report['cycles'].append(dict(cycle=cycle,sim_time=obs['sim_time'],request=request,
                                             decoded_chunk=chunk.tolist(),**prediction))
                for index,raw in enumerate(chunk[0,:args.actions_per_cycle,:7]):
                    before=obs['tcp_pos_world'].copy()
                    physical_delta=adapter.physical_delta(raw)
                    native=adapter.to_native(raw,obs)
                    obs,reward,terminated,truncated,info=env.step(native)
                    assert np.isfinite(obs['proprio']).all()
                    report['steps'].append(dict(cycle=cycle,index=index,raw_xr0_action=raw.tolist(),
                        calvin_physical_delta=physical_delta.tolist(),native_action=native.tolist(),
                        tcp_before=before.tolist(),tcp_after=obs['tcp_pos_world'].tolist(),
                        gripper_width=obs['gripper_width'],info=info))
                    if terminated or truncated:raise RuntimeError('Episode ended before completing requested smoke cycles')
                print('XR0 CYCLE',cycle,'shape',chunk.shape,'finite',np.isfinite(chunk).all(),
                      'sim_time',obs['sim_time'],'IK',info['ik_error'],flush=True)
                (out/'xr0_actions.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            accepted=sum(not s['info']['ik_failed'] for s in report['steps'])
            assert accepted>=3
            report.update(status='PASS',replan_cycles=len(report['cycles']),accepted_ik_actions=accepted,
                rejected_ik_actions=len(report['steps'])-accepted,policy_outcome='EXPECTED CROSS-DOMAIN POLICY BEHAVIOR',
                final_sim_time=obs['sim_time'],final_info=info)
        except Exception as exc:
            report.update(status='FAIL',error=type(exc).__name__,message=str(exc))
            error=exc
            traceback.print_exc()
        finally:
            env.close()
            if proc is not None:
                if proc.poll() is None:
                    try:
                        proc.stdin.write('{"stop": true}\n');proc.stdin.flush();proc.stdin.close()
                        proc.wait(timeout=15)
                    except (BrokenPipeError,subprocess.TimeoutExpired):
                        proc.terminate()
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                report['worker_returncode']=proc.returncode
            report['gpu_after_cleanup_mib']=gpu_memory()
            (out/'xr0_actions.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('XR0 FINAL',report['status'],flush=True)
    if error is not None:raise error


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--model',default=DEFAULT_MODEL)
    ap.add_argument('--model-python',default=DEFAULT_PYTHON)
    ap.add_argument('--output',default='output/xr0_smoke');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--cycles',type=int,default=3);ap.add_argument('--actions-per-cycle',type=int,default=3)
    ap.add_argument('--timeout',type=int,default=180)
    args=ap.parse_args()
    if args.worker:return worker(args)
    if not 3<=args.cycles<=5 or not 2<=args.actions_per_cycle<=5:ap.error('Smoke: 3-5 cycles, 2-5 actions/cycle')
    parent(args)
    return 0


if __name__=='__main__':sys.exit(main())
