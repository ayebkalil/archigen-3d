"""Attach explicit, artifact-scoped visual reviews without rewriting measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.furniture.tracking import get_client
from src.stress.campaign import dump
from src.stress.report import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--reviews',required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{32}',args.run_id): raise ValueError('Invalid MLflow run ID')
    directory = ROOT/'outputs/stress'/args.run_id
    data = json.loads((directory/'campaign.json').read_text(encoding='utf-8'))
    reviews = json.loads(Path(args.reviews).read_text(encoding='utf-8'))
    client = get_client()
    for review in reviews:
        record = next(r for r in data['records'] if r['run_id']==review['run_id'])
        if review['quality'] not in ('usable','degraded','unusable','not_evaluated'): raise ValueError('Invalid review quality')
        artifact = directory/review['run_id']/review.get('artifact','prepared.png')
        if artifact.resolve().parent != (directory/review['run_id']).resolve(): raise ValueError('Review artifact must belong to the run')
        review['artifact_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        review['scope'] = 'prepared_input_only; no TripoSR geometry was generated'
        review['reviewer'] = 'codex_visual_inspection'
        review['reviewed_at_utc'] = datetime.now(timezone.utc).isoformat()
        record['visual_review'] = 'reviewed_prepared_input'
        record['quality'] = review['quality']
        record['visual_review_details'] = review
        path = directory/review['run_id']/'review.json'
        dump(path,review)
        client.log_artifact(record['run_id'],str(path))
        client.set_tag(record['run_id'],'quality',review['quality'])
        client.set_tag(record['run_id'],'visual_review',record['visual_review'])
        client.set_tag(record['run_id'],'review_scope',review['scope'])
    # The original per-run record artifact remains the pre-review measurement.
    dump(directory/'campaign.json',data)
    (directory/'report.md').write_text(render(data),encoding='utf-8')
    client.log_artifact(args.run_id,str(directory/'campaign.json'))
    client.log_artifact(args.run_id,str(directory/'report.md'))
    print(directory/'report.md')


if __name__=='__main__': main()
