"""Verify terminal stress runs and their artifacts against the live MLflow API."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.furniture.tracking import get_client


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    directory=ROOT/'outputs/stress'/args.run_id
    campaign=json.loads((directory/'campaign.json').read_text(encoding='utf-8'))
    client=get_client('http://127.0.0.1:5000')
    parent=client.get_run(args.run_id)
    children=[]; token=None
    while True:
        page=client.search_runs([parent.info.experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{args.run_id}'",max_results=1000,page_token=token)
        children.extend(page);token=page.token
        if not token:break
    indexed={run.info.run_id:run for run in children}
    errors=[]
    if parent.info.status=='RUNNING':errors.append('parent still RUNNING')
    if any(run.info.status=='RUNNING' for run in children):errors.append('child still RUNNING')
    checked=0
    for record in campaign['records']:
        run=indexed.get(record['run_id'])
        if run is None:errors.append('missing child '+record['run_id']);continue
        expected='FINISHED' if record['technical_status'] in ('success','expected_rejection') else 'FAILED'
        if run.info.status!=expected:errors.append('unexpected status '+record['run_id'])
        artifacts={item.path:item for item in client.list_artifacts(record['run_id'])}
        for name in ['record.json','input.bin','trace.json','model.glb','texture.png']:
            local=directory/record['run_id']/name
            if not local.exists():continue
            if name not in artifacts:
                errors.append(f'missing artifact: {record["run_id"]}/{name}')
            elif name=='record.json':
                # Local recovery annotations can postdate the immutable measurement artifact.
                pass
            elif artifacts[name].file_size is None and local.stat().st_size==0:
                downloaded=client.download_artifacts(record['run_id'],name,str(directory/'verification-downloads'/record['run_id']))
                if Path(downloaded).stat().st_size!=0:errors.append('empty artifact mismatch '+record['run_id'])
            elif artifacts[name].file_size!=local.stat().st_size:
                errors.append(f'size mismatch: {record["run_id"]}/{name}')
        if record['technical_status']=='success' and not run.data.params.get('weights'):
            errors.append('missing weight provenance '+record['run_id'])
        checked+=1
    result={'run_id':args.run_id,'parent_status':parent.info.status,'children':len(children),
            'measured_and_probe_runs_checked':checked,'errors':errors,'tracking_warnings':campaign.get('tracking_warnings',[])}
    path=directory/'tracking-verification.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    client.log_artifact(args.run_id,str(path))
    print(json.dumps(result,indent=2))
    return int(bool(errors or result['tracking_warnings']))


if __name__=='__main__':raise SystemExit(main())
