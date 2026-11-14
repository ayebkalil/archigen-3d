"""Download a campaign summary through the existing MLflow server and render it."""
import argparse
import json
from pathlib import Path
import sys
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.furniture.tracking import get_client
from src.stress.report import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--tracking-uri',default='http://127.0.0.1:5000')
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{32}',args.run_id): raise ValueError('Invalid MLflow run ID')
    directory = ROOT/'outputs/stress'/args.run_id
    directory.mkdir(parents=True,exist_ok=True)
    client = get_client(args.tracking_uri)
    path = client.download_artifacts(args.run_id,'campaign.json',str(directory))
    campaign = json.loads(Path(path).read_text(encoding='utf-8'))
    output = directory/'report.md'
    output.write_text(render(campaign),encoding='utf-8')
    print(output)


if __name__=='__main__': main()
