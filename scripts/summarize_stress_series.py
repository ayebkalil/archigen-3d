"""Aggregate measured stress trials without mixing warmups or claiming CPU/GPU speedup."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-ids',nargs='+',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/stress/restored-series')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    campaigns=[json.loads((ROOT/'outputs/stress'/run/'campaign.json').read_text(encoding='utf-8')) for run in args.run_ids]
    records=[dict(r,phase_run_id=c['run_id']) for c in campaigns for r in c['records']]
    measured=[r for r in records if r['run_kind']=='measurement']
    grouped=defaultdict(list)
    for r in measured:grouped[(r['phase_run_id'],r['model'],r['case']['id'])].append(r)
    rows=[]
    for (phase,model,case),group in grouped.items():
        success=[r for r in group if r['technical_status']=='success']
        times=[r['metrics']['http_total_seconds'] for r in success]
        row={'phase_run_id':phase,'model':model,'case':case,'attempts':len(group),'successes':len(success),
             'technical_statuses':json.dumps(dict(Counter(r['technical_status'] for r in group))),
             'quality_statuses':json.dumps(dict(Counter(r['quality'] for r in group))),
             'reviewed_examples':sum(r.get('visual_review','not_reviewed')!='not_reviewed' for r in group),
             'http_min_s':min(times) if times else None,'http_median_s':statistics.median(times) if times else None,
             'http_max_s':max(times) if times else None,'example_run_id':group[0]['run_id']}
        keys=['input_bytes','input_width','input_height','rooms','walls','expected_rooms','synthetic_room_matched_iou_mean',
              'vertices','faces','atlas_coverage','peak_allocated_mib','peak_reserved_mib',
              'device_global_sampled_peak_mib','service_tree_rss_sampled_peak_mib','queue_seconds','inference_seconds',
              'preprocessing_seconds','vectorization_seconds','construction_3d_seconds','mesh_extraction_seconds',
              'texture_baking_seconds','export_seconds']
        for key in keys:
            values=[r['metrics'][key] for r in success if key in r['metrics']]
            if values:row[key+'_median']=statistics.median(values)
        rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (args.output/'cases.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary={'phase_run_ids':args.run_ids,'attempts_including_warmups_and_probes':len(records),
             'measured_attempts':len(measured),'models':{},'cases':rows,
             'skipped':[dict(item,phase_run_id=c['run_id']) for c in campaigns for item in c['skipped']],
             'limitations':['No CPU/GPU comparison repeated.','Only successful HTTP latencies enter min/median/max.',
                            'Quality counts mix structural checks and explicitly tagged example reviews.',
                            'Memory samples are shared service-tree RSS or global GPU samples, not per-request ownership.',
                            'No real annotated floorplan dataset or reference 3D scan; synthetic room matching only.']}
    for model in sorted({r['model'] for r in measured}):
        group=[r for r in measured if r['model']==model]
        summary['models'][model]={'attempts':len(group),'statuses':dict(Counter(r['technical_status'] for r in group)),
            'quality':dict(Counter(r['quality'] for r in group)),
            'source_sizes':sorted({(r['metrics'].get('input_width'),r['metrics'].get('input_height')) for r in group if r['metrics'].get('input_width')}),
            'observed_onnx_inputs':sorted({r['parameters'].get('observed_onnx_inputs','[]') for r in group}),
            'observed_model_inputs':sorted({r['parameters'].get('observed_model_inputs','[]') for r in group})}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','skipped')},indent=2))


if __name__=='__main__':main()
