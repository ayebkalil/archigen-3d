"""Bounded campaign controller. Run via scripts/run_stress_campaign.ps1 on Windows."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid

import httpx
import psutil
from PIL import Image

from src.furniture.tracking import get_client, code_version
from .inputs import cases
from .isolation import Job

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / '.venv-furniture-quality/Scripts/python.exe'
MIB = 1024**2


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def balanced_cases(selected):
    """Visit every transformation before spending time on its harder levels."""
    families = {}
    for case in selected:
        families.setdefault(case['family'], {}).setdefault(case['transform'], []).append(case)
    ordered = []
    for groups in families.values():
        ordered.append([group[level] for level in range(max(len(g) for g in groups.values()))
                        for group in groups.values() if level < len(group)])
    return [family[level] for level in range(max((len(g) for g in ordered), default=0))
            for family in ordered if level < len(family)]


def model_deadline(now, global_end, model, remaining, budgets, scale=1):
    reserved = sum(budgets.get(name, 0) for name in remaining) * 60 * scale
    return min(global_end - reserved, now + budgets[model] * 60 * scale)


class Tracking:
    def __init__(self, uri, parent=None):
        self.client, self.experiment = None, None
        self.warnings = []
        try:
            self.client = get_client(uri)
            experiment = self.client.get_experiment_by_name('archigen-stress-testing')
            self.experiment = experiment.experiment_id if experiment else self.client.create_experiment('archigen-stress-testing')
        except Exception as exc:
            self.warning(exc)

    def warning(self, exc):
        message = f'MLflow incomplete: {type(exc).__name__}: {exc}'
        self.warnings.append(message)
        print(message, flush=True)

    def start(self, name, parent=None, **tags):
        if self.client and self.experiment is not None:
            try:
                if parent: tags['mlflow.parentRunId'] = parent
                return self.client.create_run(self.experiment, tags={'mlflow.runName': name, **tags}).info.run_id
            except Exception as exc: self.warning(exc)
        return uuid.uuid4().hex

    def finish(self, run_id, record, directory, failed=False):
        if not self.client or self.experiment is None: return
        try:
            from mlflow.entities import Metric, Param
            params = record.get('parameters', {})
            now = int(time.time()*1000)
            metrics = record.get('metrics', {})
            self.client.log_batch(run_id, params=[Param(k,str(v)[:6000]) for k,v in params.items()],
                                  metrics=[Metric(k,float(v),now,0) for k,v in metrics.items()
                                           if isinstance(v,(int,float)) and abs(float(v)) != float('inf') and float(v)==float(v)])
            for key in ('technical_status', 'quality', 'visual_review', 'error_stage', 'run_kind', 'recovery'):
                if key in record: self.client.set_tag(run_id, key, str(record[key])[:5000])
            self.client.log_artifacts(run_id, str(directory))
        except Exception as exc: self.warning(exc)
        finally:
            try: self.client.set_terminated(run_id, 'FAILED' if failed else 'FINISHED')
            except Exception as exc: self.warning(exc)


def inventory():
    from .assets import recover_local_weights
    recovered = recover_local_weights(ROOT)
    weights = {}
    for model, paths in {
        'floorplan': ['models/saved/floorplan_unetpp/floorplan_multitask.onnx',
                      'models/saved/floorplan_unetpp/floorplan_fallback.onnx',
                      'models/saved/floorplan_unetpp/multitask_resnet34_finetuned_best.pth',
                      'models/saved/floorplan_unetpp/multitask_resnet34_best.pth',
                      'models/saved/floorplan_baseline/best.safetensors'],
        'facade': ['models/saved/generator_perceptual_ema.onnx','models/saved/generator_perceptual.onnx',
                   'models/saved/generator_perceptual_best.pth','models/saved/generator_enhanced.onnx'],
        'furniture': ['TripoSR/model.ckpt'],
    }.items():
        weights[model] = [{'path': p, 'bytes': (ROOT/p).stat().st_size, 'sha256': sha(ROOT/p)}
                          for p in paths if (ROOT/p).is_file()]
    diff = subprocess.check_output(['git','diff','HEAD','--binary'], cwd=ROOT)
    untracked = subprocess.check_output(['git','ls-files','--others','--exclude-standard'], cwd=ROOT, text=True).splitlines()
    # Include the new harness itself, absent from a tracked-files diff until committed.
    harness_hashes = {p: sha(ROOT/p) for p in untracked if (ROOT/p).is_file() and p.startswith(('src/stress/','scripts/','configs/'))}
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'], text=True, timeout=10).strip()
    except Exception as exc: gpu = str(exc)
    import importlib.metadata
    packages = {}
    for package in ('torch','onnxruntime','mlflow-skinny','numpy','pillow','trimesh','transformers','psutil'):
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    triposr = {}
    if (ROOT/'TripoSR/.git').exists():
        triposr['revision'] = subprocess.check_output(['git','-C',str(ROOT/'TripoSR'),'rev-parse','HEAD'],text=True).strip()
        triposr['dirty'] = bool(subprocess.check_output(['git','-C',str(ROOT/'TripoSR'),'status','--porcelain'],text=True).strip())
    if (ROOT/'TripoSR/config.yaml').exists(): triposr['config_sha256'] = sha(ROOT/'TripoSR/config.yaml')
    return {'weights': weights, 'triposr_source':triposr,'code': {**code_version(), 'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
                                       'untracked_harness_hashes': harness_hashes}, 'packages': packages,
            'gpu': gpu, 'ram_available_mib': psutil.virtual_memory().available/MIB,'local_weight_recovery':recovered,
            'local_dvc_cache': (ROOT/'.dvc/cache').exists(),
            'ground_truth_available': any((ROOT/'data/raw').rglob('model.svg'))}


class Sampler:
    def __init__(self, child):
        self.child, self.stop_event = child, threading.Event()
        self.rss, self.gpu, self.samples = [], [], []
    def __enter__(self):
        def sample():
            while not self.stop_event.is_set():
                item = {'time': time.time(), 'available_ram_mib': psutil.virtual_memory().available/MIB}
                try:
                    process = psutil.Process(self.child.pid)
                    item['service_tree_rss_mib'] = sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())/MIB
                    self.rss.append(item['service_tree_rss_mib'])
                except psutil.Error: pass
                try:
                    used = float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0])
                    self.gpu.append(used); item['device_global_mib'] = used
                except Exception: pass
                self.samples.append(item)
                self.stop_event.wait(.5)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join(timeout=5)
    def metrics(self):
        result = {}
        if self.rss: result['service_tree_rss_sampled_peak_mib'] = max(self.rss)
        if self.gpu: result['device_global_sampled_peak_mib'] = max(self.gpu)
        return result


class Service:
    def __init__(self, campaign, model):
        self.campaign, self.model, self.job, self.child = campaign, model, None, None
    def start(self):
        self.stop()
        self.job = Job(self.campaign.memory_bytes)
        self.log = (self.campaign.directory/f'{self.model}-service.log').open('ab')
        env = self.campaign.env.copy()
        env['ARCHIGEN_STRESS_MODEL'] = self.model
        python = PYTHON if self.model.startswith('furniture') else ROOT/'.venv/Scripts/python.exe'
        self.child = self.job.start(python, 'uvicorn', ['src.stress.server:app','--host','127.0.0.1','--port','18001'],
                                    cwd=ROOT, env=env, stdout=self.log)
        deadline = min(time.monotonic()+120, self.campaign.deadline-30)
        while time.monotonic() < deadline:
            if psutil.virtual_memory().available/MIB < self.campaign.config['min_available_ram_mib']:
                self.stop()
                raise RuntimeError('RAM guard stopped service initialization')
            if self.child.poll() is not None:
                raise RuntimeError(f'Isolated service exited with code {self.child.returncode} (0x{self.child.returncode & 0xffffffff:08x}); see service log')
            try:
                response = httpx.get('http://127.0.0.1:18001/health',timeout=2,trust_env=False)
                if response.is_success and response.json().get('stress_only'):
                    self.health = response.json()
                    if not self.job.contains(self.health['process_id']):
                        raise RuntimeError('Service interpreter is outside the campaign Job Object; no workload admitted')
                    self.health['verified_job_membership'] = True
                    return self.health
            except httpx.HTTPError: pass
            time.sleep(.5)
        raise TimeoutError('Isolated service startup deadline')
    def stop(self):
        if self.job: self.job.close(); self.job = None
        if getattr(self,'log',None): self.log.close(); self.log = None


class Campaign:
    def __init__(self, config, minutes, uri):
        if config['gpu_guard_mib']!=3500:
            raise ValueError('The existing TripoSR guard is fixed at 3500 MiB; the manifest cannot silently change it')
        self.config, self.started = config, time.monotonic()
        self.deadline = self.started+minutes*60
        self.measure_end = self.started+minutes*60*23/30
        self.global_measure_end = self.measure_end
        self.budget_scale = config.get('model_budget_scale', minutes / 30)
        self.setup_end = self.started+minutes*60*5/30
        self.tracking = Tracking(uri)
        self.run_id = self.tracking.start('stress-campaign', run_kind='stress_campaign')
        self.directory = ROOT/'outputs/stress'/self.run_id
        self.directory.mkdir(parents=True)
        dump(self.directory/'manifest.json',config)
        self.records, self.skipped, self.services = [], [], []
        self.validation_lock = threading.Lock()
        self.memory_bytes = int(min(config['max_job_mib']*MIB, psutil.virtual_memory().available*config['memory_available_fraction']))
        self.env = {**os.environ, 'ARCHIGEN_MLFLOW_ENABLED':'0', 'ARCHIGEN_FURNITURE_DETAIL':config['furniture_detail'],
                    'ARCHIGEN_FLOORPLAN_BACKEND':'onnx', 'ARCHIGEN_FACADE_EXPERIMENT':'',
                    'ARCHIGEN_STRESS_DIRECTORY':str(self.directory), 'HF_HOME':str(ROOT/'outputs/runtime/huggingface'),
                    'U2NET_HOME':str(ROOT/'outputs/runtime/rembg'), 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
                    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}

    def room(self, seconds):
        return time.monotonic()+seconds < self.measure_end and psutil.virtual_memory().available/MIB >= self.config['min_available_ram_mib']

    def build_input(self, case, source=None):
        directory = self.directory/'inputs'/case['id']
        directory.mkdir(parents=True, exist_ok=True)
        output = directory/'input.bin'
        with Job(self.memory_bytes) as job, (directory/'generator.log').open('wb') as log:
            child = job.start(PYTHON,'src.stress.inputs',['--case',json.dumps(case),'--source',source or ROOT/self.config['furniture_image'],
                              '--seed',self.config['seed'],'--output',output], cwd=ROOT, env=self.env, stdout=log)
            child.wait(timeout=min(30,max(1,self.measure_end-time.monotonic())))
            if child.returncode: raise RuntimeError('Input generation failed in bounded process')
        return output

    def attempt(self, service, case, input_path, iteration, kind='measurement', concurrency=1):
        run = self.tracking.start(f"{service.model}-{case['id']}-{iteration}",self.run_id,run_kind=kind,model=service.model)
        directory = self.directory/run
        directory.mkdir(exist_ok=True)
        record = {'run_id':run,'model':service.model,'case':case,'iteration':iteration,'run_kind':kind,
                  'technical_status':'error','quality':'not_evaluated','visual_review':'not_reviewed',
                  'parameters':{'model':service.model,'iteration':iteration,'concurrency':concurrency,
                                'seed':self.config['seed'],'transform':json.dumps(case,sort_keys=True),
                                'preset':self.config['furniture_detail'] if service.model=='furniture' else 'not_applicable', 'run_kind':kind,
                                'job_memory_limit_mib':self.memory_bytes/MIB,
                                'min_available_ram_mib':self.config['min_available_ram_mib'],
                                'gpu_guard_mib':self.config['gpu_guard_mib'],
                                'seed_scope':'input transformations; no claim about third-party packing determinism',
                                'git_revision': self.inv['code'].get('git_revision'),
                                'tracked_diff_sha256':self.inv['code']['tracked_diff_sha256'],
                                'weights':json.dumps(self.inv['weights'][service.model]),
                                'software':json.dumps(self.inv['packages'])},'metrics':{}}
        record['parameters']['service_runtime'] = json.dumps(getattr(service,'health',{}))
        expected_alpha = None
        http_started = None
        try:
            data = input_path.read_bytes()
            if service.model=='facade' and not case.get('semantic'):
                data = json.dumps(case['payload'],sort_keys=True,separators=(',',':')).encode()
                (directory/'input.json').write_bytes(data)
            (directory/'input.bin').write_bytes(data)
            expected_alpha = input_path.with_name('expected-alpha.png')
            if expected_alpha.exists():
                import shutil
                shutil.copyfile(expected_alpha,directory/'expected-alpha.png')
            expected_geometry = input_path.with_name('expected-geometry.json')
            if expected_geometry.exists():
                import shutil
                shutil.copyfile(expected_geometry,directory/'expected-geometry.json')
            record['parameters']['input_sha256'] = hashlib.sha256(data).hexdigest()
            record['metrics']['input_bytes'] = len(data)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    record['metrics'].update(input_width=image.width,input_height=image.height)
            except Exception: pass  # Header probe only; invalid content is still sent to the service.
            headers = {'x-archigen-stress-id':run}
            timeout = self.config['timeouts_seconds'][service.model]*concurrency+30
            started = time.perf_counter()
            http_started = started
            record['_http_started'] = started
            if service.model.startswith('furniture'):
                payload = {'image_base64':base64.b64encode(data).decode(),'name':'Stress chair','category':
                           'unknown_stress_category' if case['transform']=='unusual_category' else 'chair',
                           'target_height_m':.85,'remove_bg':True}
                record['parameters'].update(category=payload['category'],target_height_m=.85,remove_bg=True,
                    execution_scope='preprocessing_only' if service.model=='furniture_preprocessing' else 'neural_reconstruction')
                response = httpx.post('http://127.0.0.1:18001/furniture/generate-3d',json=payload,headers=headers,timeout=timeout,trust_env=False)
            elif service.model == 'floorplan':
                response = httpx.post('http://127.0.0.1:18001/predict/floorplan', files={'file':('input.png',data,'image/png')},
                                      data={'scale_override':'.03'},headers=headers,timeout=timeout,trust_env=False)
            elif case.get('semantic'):
                response = httpx.post('http://127.0.0.1:18001/stress/facade-semantic',files={'file':('input.png',data,'image/png')},
                                      headers=headers,timeout=timeout,trust_env=False)
            else:
                response = httpx.post('http://127.0.0.1:18001/facade/generate',content=data,
                                      headers={**headers,'content-type':'application/json'},timeout=timeout,trust_env=False)
            record['metrics']['http_total_seconds'] = time.perf_counter()-started
            record['_http_finished'] = time.perf_counter()
            record['metrics']['http_status'] = response.status_code
            try: result = response.json()
            except ValueError: result = {'raw_response':response.text[:10000]}
            record['technical_status'] = ('success' if response.is_success else
                'expected_rejection' if response.status_code in (400,415,422) and case.get('expected_rejection') else 'error')
            if not response.is_success: record['error'] = result
            if response.is_success:
                for key in ('density_resolution','texture_resolution','precision','checkpoint_loading'):
                    if (result.get('quality') or {}).get(key) is not None:
                        record['parameters'][key] = result['quality'][key]
                checked = ({'quality':'not_evaluated','issues':[],'metrics':{},
                            'limitation':'Preprocessing only; no neural reconstruction or geometry evaluation'}
                           if service.model=='furniture_preprocessing' else self.validate_output(service.model,result,directory))
                record['quality'] = checked['quality']
                record['evaluation'] = checked
                record['metrics'].update(checked['metrics'])
                # A blank input accepted with geometry is a documented semantic anomaly.
                if service.model=='furniture' and case['transform'] in ('uniform','transparent'):
                    record['quality'] = 'unusable'
                    record['evaluation']['issues'].append('geometry_from_input_without_object')
            # Avoid duplicating huge GLB/base64 strings already stored as binary artifacts.
            dump(directory/'response.json',{k:v for k,v in result.items() if 'base64' not in k})
        except httpx.TimeoutException as exc:
            record.update(technical_status='timeout',error=str(exc),traceback=traceback.format_exc())
        except httpx.TransportError as exc:
            record.update(technical_status='crash' if service.child.poll() is not None else 'transport_error',error=str(exc),traceback=traceback.format_exc())
        except Exception as exc:
            record.update(technical_status='error',error=str(exc),error_stage='output_validation',traceback=traceback.format_exc())
        if http_started is not None and 'http_total_seconds' not in record['metrics']:
            record['metrics']['http_failure_detection_seconds'] = time.perf_counter()-http_started
        trace_path = directory/'trace.json'
        if trace_path.exists():
            trace = json.loads(trace_path.read_text(encoding='utf-8'))
            record['error_stage'] = trace.get('error_stage',record.get('error_stage',
                'api_validation' if record['technical_status']=='expected_rejection' else
                'none' if record['technical_status']=='success' else 'indeterminate'))
            record['metrics'].update({k+'_seconds':v for k,v in trace.get('stages_seconds',{}).items()})
            for key in ('queue_seconds','service_total_seconds'): 
                if key in trace: record['metrics'][key] = trace[key]
            record['parameters']['observed_model_inputs'] = json.dumps(trace.get('model_inputs',[]))
            record['parameters']['observed_onnx_inputs'] = json.dumps(trace.get('onnx_inputs',[]))
            record['parameters']['observed_onnx_input_dtypes'] = json.dumps(trace.get('onnx_input_dtypes',[]))
            record['parameters']['observed_onnx_models'] = json.dumps(trace.get('onnx_models',[]))
            record['metrics']['observed_encoder_calls'] = len(trace.get('model_inputs',[]))
            record['metrics']['observed_onnx_calls'] = len(trace.get('onnx_inputs',[]))
            if trace.get('prepared_dimensions'):
                record['metrics'].update(prepared_width=trace['prepared_dimensions'][0],prepared_height=trace['prepared_dimensions'][1])
        raw_mask = directory/'raw-mask.png'
        if expected_alpha is not None and expected_alpha.exists() and raw_mask.exists():
            try:
                record['metrics'].update(self.validate_output(service.model,None,directory,mode='mask'))
            except Exception as exc:
                record['mask_evaluation_error'] = repr(exc)
            record['parameters']['mask_ground_truth'] = 'source alpha retained before synthetic compositing/JPEG; segmentation only, not 3D fidelity'
        else: record.setdefault('error_stage','indeterminate')
        if record.get('technical_status')=='error' and 'out of memory' in str(record.get('error','')).lower():
            record['technical_status'] = 'out_of_memory'
        if getattr(service,'watchdog_reason',None):
            record['technical_status'] = service.watchdog_reason
            record['watchdog_observation'] = getattr(service,'watchdog_observation',{})
        record['_directory'] = str(directory)
        return record

    def validate_output(self, model, result, directory, mode='output'):
        # Validation itself can receive huge masks or meshes; bound and serialize it
        # rather than allocating multiple copies inside the campaign controller.
        with self.validation_lock, Job(min(self.memory_bytes,2*1024**3)) as job:
            if mode=='output': dump(directory/'validation-input.json',result)
            with (directory/'validation.log').open('wb') as log:
                child = job.start(PYTHON,'src.stress.evaluate',['--mode',mode,'--model',model,'--directory',directory],
                                  cwd=ROOT,env=self.env,stdout=log)
                child.wait(timeout=60)
                if child.returncode: raise RuntimeError(f'Isolated {mode} validation failed with code {child.returncode}')
            if mode=='output': (directory/'validation-input.json').unlink(missing_ok=True)
            return json.loads((directory/('mask-metrics.json' if mode=='mask' else 'evaluation.json')).read_text(encoding='utf-8'))

    def complete(self, records, sampler, batch_seconds):
        shared = sampler.metrics()
        completed = [r for r in records if '_http_finished' in r]
        http_span = (max(r['_http_finished'] for r in completed)-min(r['_http_started'] for r in completed)) if completed else None
        for record in records:
            record['metrics'].update(shared,batch_seconds=batch_seconds,
                                     validated_batch_throughput_success_per_second=sum(r['technical_status']=='success' for r in records)/max(batch_seconds,.001))
            if http_span is not None:
                record['metrics'].update(http_batch_seconds=http_span,
                    batch_throughput_success_per_second=sum(r['technical_status']=='success' for r in completed)/max(http_span,.001))
            record.pop('_http_started',None);record.pop('_http_finished',None)
            record['parameters']['memory_attribution'] = 'shared service tree and global GPU; not per-request allocations'
            directory = Path(record.pop('_directory'))
            dump(directory/'memory-samples.json',sampler.samples)
            dump(directory/'record.json',record)
            self.records.append(record)
            self.tracking.finish(record['run_id'],record,directory,
                                 failed=record['technical_status'] not in ('success','expected_rejection'))
            print(json.dumps({'run':record['run_id'],'case':record['case']['id'],'iteration':record['iteration'],
                              'status':record['technical_status'],'quality':record['quality'],
                              'seconds':record['metrics'].get('http_total_seconds')}),flush=True)
        self.persist()

    def batch(self, service, case, path, count=1, kind='measurement', iteration=0):
        service.watchdog_reason = None
        started = time.perf_counter()
        with Sampler(service.child) as sampler, ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(self.attempt,service,case,path,iteration+i,kind,count) for i in range(count)]
            # Independent watchdog: kills service calculations, not merely the HTTP clients.
            ceiling = case.get('watchdog_seconds', self.config['timeouts_seconds']['concurrent_furniture'] if count>1 else self.config['timeouts_seconds'][service.model])
            deadline = time.monotonic()+min(ceiling,max(1,self.deadline-time.monotonic()-60))
            while not all(f.done() for f in futures):
                available_mib = psutil.virtual_memory().available/MIB
                if time.monotonic() >= deadline or available_mib < self.config['min_available_ram_mib']:
                    service.watchdog_reason = 'timeout' if time.monotonic()>=deadline else 'ram_guard_stop'
                    service.watchdog_observation = {'available_ram_mib':available_mib,'elapsed_seconds':time.perf_counter()-started}
                    service.stop()
                    break
                time.sleep(.1)
            records = [f.result() for f in futures]
        self.complete(records,sampler,time.perf_counter()-started)
        return records

    def persist(self):
        dump(self.directory/'campaign.json',{'run_id':self.run_id,'inventory':self.inv,'manifest':self.config,
             'records':self.records,'skipped':self.skipped,'tracking_warnings':self.tracking.warnings,
             'elapsed_seconds':time.monotonic()-self.started,'memory_limit_mib':self.memory_bytes/MIB})

    def recover(self, service, baseline, path):
        if time.monotonic()+self.config['timeouts_seconds'][service.model] > self.deadline-60:
            self.skipped.append({'model':service.model,'case':'recovery','reason':'insufficient remaining budget'})
            return False
        before = self.batch(service,baseline,path,kind='recovery_probe')[0]
        if before['technical_status']=='success':
            before['recovery'] = 'spontaneous'
            self.tag_recovery(before)
            self.persist(); return True
        try:
            service.start()
            after = self.batch(service,baseline,path,kind='recovery_probe_after_restart')[0]
            after['recovery'] = 'after_restart' if after['technical_status']=='success' else 'persistent_failure'
            self.tag_recovery(after)
            self.persist(); return after['technical_status']=='success'
        except Exception as exc:
            self.skipped.append({'model':service.model,'case':'recovery','reason':str(exc)})
            return False

    def tag_recovery(self, record):
        dump(self.directory/record['run_id']/'record.json',record)
        if self.tracking.client:
            try:
                self.tracking.client.set_tag(record['run_id'],'recovery',record['recovery'])
                self.tracking.client.log_artifact(record['run_id'],str(self.directory/record['run_id']/'record.json'))
            except Exception as exc: self.tracking.warning(exc)

    def run(self):
        self.archive_provenance()
        self.inv = inventory()
        dump(self.directory/'inventory.json',self.inv)
        self.persist()
        models = list(self.config.get('models', ['floorplan','facade','furniture']))
        if self.config.get('components_only'):
            models = ['furniture_preprocessing']
        budgets = self.config.get('model_minutes', {'floorplan':6, 'facade':4, 'furniture':8,
                                                   'furniture_preprocessing':2})
        for model_index, model in enumerate(models):
            self.measure_end = model_deadline(time.monotonic(), self.global_measure_end, model,
                models[model_index+1:], budgets, self.budget_scale)
            if model=='furniture_preprocessing':
                path = ROOT/'outputs/runtime/rembg/u2netp.onnx'
                self.inv['weights'][model] = [{'path':str(path.relative_to(ROOT)), 'sha256':sha(path), 'bytes':path.stat().st_size}] if path.is_file() else []
            if not self.inv['weights'][model]:
                self.skipped.append({'model':model,'case':'all_neural_tests','reason':'weights_absent; local DVC cache unavailable; mock excluded'})
                run = self.tracking.start(model+'-blocked',self.run_id,run_kind='availability_check',model=model)
                directory = self.directory/run; directory.mkdir()
                blocked = {'technical_status':'blocked','parameters':{'model':model,'reason':self.skipped[-1]['reason']}}
                dump(directory/'record.json',blocked)
                self.tracking.finish(run,blocked,directory)
                if model=='furniture' and self.config.get('component_tests_when_neural_blocked'):
                    models.append('furniture_preprocessing')
                continue
            service = Service(self,model); self.services.append(service)
            setup_stage = 'input_setup'
            try:
                source = ROOT/self.config['furniture_image'] if model.startswith('furniture') else next((ROOT/self.config['plan_directory']).glob('*.png'))
                baseline = dict(id='reference',family='reference',transform='identity')
                if model=='facade': baseline['payload'] = {'style':'Modern Minimalist','num_floors':1}
                setup_stage = 'model_initialization'
                health = service.start(); dump(self.directory/f'{model}-health.json',health)
                setup_stage = 'warmup'
                self.batch(service,baseline,source,kind='warmup')
                if model.startswith('furniture'):
                    opaque = {'id':'opaque-warmup','family':'reference','transform':'resize','value':512,'format':'jpeg'}
                    opaque_path = self.build_input(opaque, source)
                    self.batch(service,opaque,opaque_path,kind='warmup')
                if time.monotonic()>self.setup_end:
                    self.skipped.append({'model':model,'case':'setup_budget','reason':'startup exceeded setup allocation; measurement window shortened'})
                all_cases = balanced_cases(cases(self.config,'furniture' if model=='furniture_preprocessing' else model))
                selected_ids = self.config.get('include_cases_by_model', {}).get(model, self.config.get('include_cases'))
                if selected_ids:
                    all_cases = balanced_cases([c for c in all_cases if c['id'] in selected_ids])
                # Baseline included in measured results; warmup is separate.
                all_cases.insert(0, baseline)
                for case in all_cases:
                    if not self.room(self.config['timeouts_seconds'][model]):
                        self.skipped.append({'model':model,'case':case,'reason':'budget or RAM admission limit'})
                        continue
                    try: path = source if case==baseline else self.build_input(case, source)
                    except Exception as exc:
                        self.skipped.append({'model':model,'case':case,'reason':'bounded input generator: '+str(exc)})
                        continue
                    for iteration in range(self.config['repetitions']):
                        if not self.room(self.config['timeouts_seconds'][model]):
                            self.skipped.append({'model':model,'case':case,'iteration':iteration,'reason':'budget or RAM admission limit'}); break
                        record = self.batch(service,case,path,iteration=iteration)[0]
                        if record['technical_status']!='success':
                            if not self.recover(service,baseline,source): break
                for count in self.config['concurrency']:
                    reference_times = [r['metrics']['http_total_seconds'] for r in self.records
                                       if r['model']==model and r['case']['id']=='reference' and r['technical_status']=='success']
                    timeout = min(self.config['timeouts_seconds'][model]*count,
                                  max(30,(max(reference_times)*3*count+15) if reference_times else 180*count))
                    if not self.room(timeout):
                        self.skipped.append({'model':model,'case':f'concurrency_{count}','reason':'budget or RAM admission limit'}); continue
                    for repeat in range(self.config['repetitions']):
                        if not self.room(timeout): break
                        records = self.batch(service,{**baseline,'id':f'concurrency_{count}','family':'concurrency',
                                                      'watchdog_seconds':timeout},source,count,iteration=repeat*count)
                        if any(r['technical_status']!='success' for r in records) and not self.recover(service,baseline,source): break
                iteration = 0
                while self.room(self.config['timeouts_seconds'][model]) and iteration<self.config.get('endurance_max_attempts',30):
                    record = self.batch(service,{**baseline,'id':'endurance','family':'endurance'},source,iteration=iteration)[0]
                    iteration += 1
                    if record['technical_status']!='success' and not self.recover(service,baseline,source): break
            except Exception as exc:
                self.skipped.append({'model':model,'reason':str(exc),'traceback':traceback.format_exc()})
                run = self.tracking.start(model+'-service-failure',self.run_id,run_kind='service_failure',model=model)
                directory = self.directory/run; directory.mkdir()
                failure = {'technical_status':'error','error_stage':setup_stage,
                           'parameters':{'model':model,'job_limit_mib':self.memory_bytes/MIB,'checkpoint_loading':'meta_assign_strict_no_mmap'},
                           'error':str(exc),'traceback':traceback.format_exc()}
                dump(directory/'record.json',failure)
                self.tracking.finish(run,failure,directory,failed=True)
                if model=='furniture' and self.config.get('component_tests_when_neural_blocked'):
                    print('Neural service blocked; scheduling explicitly labeled CPU preprocessing-only component tests.',flush=True)
                    models.append('furniture_preprocessing')
            finally: service.stop(); self.persist()

    def archive_provenance(self):
        import shutil
        archive = self.directory/'summary/provenance'
        archive.mkdir(parents=True,exist_ok=True)
        paths = [* (ROOT/'src').rglob('*.py'), * (ROOT/'backend/app').glob('*.py'),
                 * (ROOT/'TripoSR/tsr').rglob('*.py'),ROOT/'TripoSR/config.yaml',
                 ROOT/'scripts/run_stress_campaign.ps1',ROOT/'scripts/report_stress_campaign.py',
                 ROOT/'scripts/review_stress_campaign.py', ROOT/'requirements-stress.txt',
                 ROOT/'requirements-furniture-quality.txt',ROOT/'backend/requirements.txt']
        hashes = {}
        for path in paths:
            if not path.is_file(): continue
            relative = path.relative_to(ROOT)
            target = archive/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            hashes[str(relative)] = sha(target)
        (archive/'tracked-working-tree.diff').write_bytes(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT))
        dump(archive/'sha256.json',hashes)

    def finish(self, error=None):
        for service in self.services: service.stop()
        if not hasattr(self,'inv'): self.inv = {'error':'inventory interrupted'}
        self.persist()
        from .report import render
        report = render(json.loads((self.directory/'campaign.json').read_text(encoding='utf-8')))
        (self.directory/'report.md').write_text(report,encoding='utf-8')
        # Only parent-level files; children already have their own artifacts.
        parent_dir = self.directory/'summary'; parent_dir.mkdir(exist_ok=True)
        import shutil
        for name in ('campaign.json','report.md','inventory.json','manifest.json'):
            if (self.directory/name).exists(): shutil.copyfile(self.directory/name,parent_dir/name)
        for path in self.directory.glob('*-service.log'): shutil.copyfile(path,parent_dir/path.name)
        self.tracking.finish(self.run_id,{'technical_status':'interrupted' if error else 'completed',
             'parameters':{'budget_minutes':(self.deadline-self.started)/60,'comparison_cpu_gpu':'not_repeated'},
             'metrics':{'attempts':len(self.records),'skipped':len(self.skipped),'elapsed_seconds':time.monotonic()-self.started}},parent_dir,failed=bool(error))
        print(f'CAMPAIGN_RUN_ID={self.run_id}\nREPORT={self.directory / "report.md"}',flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget-minutes',type=float,default=30)
    parser.add_argument('--manifest',default='configs/stress_campaign.json')
    parser.add_argument('--tracking-uri',default='http://127.0.0.1:5000')
    parser.add_argument('--components-only',action='store_true',help='Explicit CPU preprocessing component tests; no neural reconstruction')
    args = parser.parse_args()
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',18001))==0: raise RuntimeError('Test port 18001 already occupied; no process stopped')
    config = json.loads((ROOT/args.manifest).read_text(encoding='utf-8'))
    if config.get('extends'):
        config = {**json.loads((ROOT/config['extends']).read_text(encoding='utf-8')),**config}
    config['components_only'] = args.components_only
    campaign = Campaign(config,args.budget_minutes,args.tracking_uri)
    error = None
    try: campaign.run()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally: campaign.finish(error)


if __name__=='__main__': main()
