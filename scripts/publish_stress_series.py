"""Publish an aggregate campaign parent; generation runs remain in their recorded phases."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.furniture.tracking import get_client


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'outputs/stress/restored-series')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();directory=args.directory
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    client=get_client('http://127.0.0.1:5000')
    phases=[client.get_run(run_id) for run_id in summary['phase_run_ids']]
    if any(p.info.status=='RUNNING' for p in phases):raise RuntimeError('Phase is still running')
    metadata_path=directory/'summary-run.json'
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        if metadata['phase_run_ids']!=summary['phase_run_ids']:raise ValueError('Use a fresh output directory for a different series')
        run_id=metadata['run_id']
    else:
        run_id=client.create_run(phases[0].info.experiment_id,start_time=min(p.info.start_time for p in phases),tags={
            'mlflow.runName':'stress-restored-weights-2026-11-14','run_kind':'campaign_summary',
            'comparison_cpu_gpu':'not_repeated','budget_minutes':'30'}).info.run_id
        metadata={'run_id':run_id,'phase_run_ids':summary['phase_run_ids']}
        metadata_path.write_text(json.dumps(metadata,indent=2))
    for phase in phases:client.set_tag(phase.info.run_id,'mlflow.parentRunId',run_id)
    for key in ['attempts_including_warmups_and_probes','measured_attempts']:
        client.log_metric(run_id,key,summary[key])
    started=min(p.info.start_time for p in phases)
    elapsed=(time.time()*1000-started)/1000
    client.log_metric(run_id,'wall_elapsed_to_publication_seconds',elapsed)
    client.set_tag(run_id,'technical_status','completed_with_observed_model_and_api_limits')
    client.log_artifacts(run_id,str(directory))
    client.log_artifact(run_id,str(args.report))
    for name in ['render_stress_mesh.py','render_stress_floorplan.py','review_stress_campaign.py',
                 'verify_stress_tracking.py','summarize_stress_series.py','publish_stress_series.py']:
        client.log_artifact(run_id,str(ROOT/'scripts'/name),artifact_path='reproduction')
    for name in ['stress_neural_restored.json','stress_architecture_restored.json','stress_final_boundaries.json','stress_furniture_large_images.json']:
        client.log_artifact(run_id,str(ROOT/'configs'/name),artifact_path='reproduction')
    client.set_terminated(run_id,'FINISHED')
    print(f'SUMMARY_RUN_ID={run_id}\nhttp://127.0.0.1:5000/#/experiments/{phases[0].info.experiment_id}/runs/{run_id}\nElapsed to publication: {elapsed:.1f}s')


if __name__=='__main__':main()
