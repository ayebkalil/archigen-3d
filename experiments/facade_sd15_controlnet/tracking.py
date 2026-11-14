"""MLflow tracking plus portable events that can be imported into local MLflow."""
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlsplit, urlunsplit

from .data import write_json


class RunTracker:
    def __init__(self, output, experiment_name, params, tags, tracking_uri=None, resume_run_id=None):
        import mlflow
        from mlflow.tracking import MlflowClient
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.uri = tracking_uri or 'sqlite:///' + (self.output / 'mlflow.db').as_posix()
        self.client = MlflowClient(tracking_uri=self.uri)
        experiment = self.client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
            experiment_name, artifact_location=(self.output / 'mlartifacts').as_uri() if not tracking_uri else None)
        if resume_run_id:
            run = self.client.get_run(resume_run_id)
            if run.info.experiment_id != experiment_id:
                raise ValueError('Resume MLflow experiment differs')
        else:
            run = self.client.create_run(experiment_id, tags={k: str(v) for k, v in tags.items()})
        self.run_id = run.info.run_id
        self.events = self.output / 'tracking-events.jsonl'
        self.artifacts = self.output / 'portable-artifacts'
        if not resume_run_id and self.events.exists():
            previous = next(json.loads(line) for line in self.events.read_text(encoding='utf-8').splitlines()
                            if line.strip() and json.loads(line).get('type') == 'run')
            history = (self.output / 'tracking-history' / previous['run_id']).resolve()
            if not history.is_relative_to(self.output): raise ValueError('Invalid tracking history path')
            history.mkdir(parents=True, exist_ok=True)
            self.events.replace(history / 'tracking-events.jsonl')
            if self.artifacts.exists(): self.artifacts.rename(history / 'portable-artifacts')
        self.artifacts.mkdir(exist_ok=True)
        if not resume_run_id:
            self._event({'type': 'run', 'run_id': self.run_id, 'experiment_name': experiment_name,
                         'start_time': run.info.start_time, 'tags': tags})
            for key, value in params.items():
                text = str(value)
                self.client.log_param(self.run_id, key, text)
                self._event({'type': 'param', 'key': key, 'value': text})
        self.client.set_tag(self.run_id, 'mlflow.note.content', 'Kaggle experiment; download portable events and model export for local use.')
        parsed = urlsplit(self.uri)
        safe_uri = urlunsplit((parsed.scheme, parsed.netloc.rsplit('@', 1)[-1], parsed.path, '', '')) if parsed.scheme in {'http', 'https'} else self.uri
        write_json(self.output / 'tracking-run.json', {'run_id': self.run_id, 'tracking_uri': safe_uri})

    def _event(self, event):
        with self.events.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')

    def metrics(self, values, step):
        from mlflow.entities import Metric
        timestamp = int(time.time() * 1000)
        self.client.log_batch(self.run_id, metrics=[Metric(k, float(v), timestamp, int(step)) for k, v in values.items()])
        for key, value in values.items():
            self._event({'type': 'metric', 'key': key, 'value': float(value), 'step': int(step), 'timestamp': timestamp})

    def artifact(self, path, folder='reports'):
        path = Path(path)
        target = self.artifacts / folder / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target.resolve(): shutil.copyfile(path, target)
        self.client.log_artifact(self.run_id, str(target), artifact_path=folder)

    def finish(self, status='FINISHED'):
        self.client.set_terminated(self.run_id, status=status)
        self._event({'type': 'status', 'status': status})


def import_tracking(bundle, tracking_uri):
    """Replay histories and portable artifacts; never transplant Kaggle absolute paths."""
    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient
    root = Path(bundle).resolve()
    events = [json.loads(line) for line in (root / 'tracking-events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    info = next(event for event in events if event['type'] == 'run')
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(info['experiment_name'])
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(info['experiment_name'])
    runs = client.search_runs([experiment_id], filter_string=f"tags.kaggle_source_run_id = '{info['run_id']}'")
    run = runs[0] if runs else client.create_run(experiment_id, start_time=info['start_time'],
        tags={**{k: str(v) for k, v in info['tags'].items()}, 'kaggle_source_run_id': info['run_id']})
    if run.data.tags.get('portable_import_complete') == 'true': return run.info.run_id
    run_id = run.info.run_id
    seen = {}
    for event in events:
        if event['type'] == 'param': client.log_param(run_id, event['key'], event['value'])
        elif event['type'] == 'metric':
            key = event['key']
            if key not in seen:
                seen[key] = {(m.step, m.timestamp, m.value) for m in client.get_metric_history(run_id, key)}
            signature = (event['step'], event['timestamp'], event['value'])
            if signature not in seen[key]:
                client.log_metric(run_id, key, event['value'], timestamp=event['timestamp'], step=event['step'])
                seen[key].add(signature)
    if (root / 'portable-artifacts').exists(): client.log_artifacts(run_id, str(root / 'portable-artifacts'))
    statuses = [event['status'] for event in events if event['type'] == 'status']
    status = statuses[-1] if statuses else 'KILLED'
    client.set_terminated(run_id, status=status)
    # An interrupted download can be imported again after its remaining events arrive.
    client.set_tag(run_id, 'portable_import_complete', 'true' if status == 'FINISHED' else 'false')
    return run_id
