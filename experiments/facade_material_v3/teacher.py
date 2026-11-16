"""Train on training photos only; validate and gate semantic feedback separately."""
import argparse
import json
import os
from pathlib import Path

from experiments.facade_sd15_controlnet.data import digest, write_json
from experiments.facade_sd15_controlnet.tracking import RunTracker
from .architecture import make_teacher
from .data import MaterialDataset


def segmentation_scores(confusion):
    import numpy as np
    c = np.asarray(confusion, dtype=np.float64)
    union = c.sum(0) + c.sum(1) - c.diagonal()
    iou = np.divide(c.diagonal(), union, out=np.zeros(6), where=union > 0)
    tp = c[2:4, 2:4].sum(); total = c[2:4].sum() + c[:, 2:4].sum() - tp
    return {'miou': float(iou[union > 0].mean()), 'wall_iou': float(iou[1]),
        'window_iou': float(iou[2]), 'door_iou': float(iou[3]),
        'opening_iou': float(tp / total) if total else 1.,
        'wall_false_opening_rate': float(c[1, 2:4].sum() / max(c[1].sum(), 1)),
        'class_iou': iou.tolist(), 'confusion': c.astype(int).tolist()}


def teacher_accepted(scores, config):
    """One shared, unchanged acceptance gate for selection and downstream use."""
    return (scores['miou'] >= config['teacher_min_miou']
        and scores['opening_iou'] >= config['teacher_min_opening_iou']
        and scores['door_iou'] >= config['teacher_min_door_iou']
        and scores['wall_false_opening_rate'] <= config['teacher_max_wall_fp'])


def teacher_selection_key(scores, config):
    # A high-recall checkpoint must never displace one that meets precision.
    overlap = .5*scores['window_iou'] + .3*scores['door_iou'] + .2*scores['wall_iou']
    return (int(teacher_accepted(scores, config)), overlap - scores['wall_false_opening_rate'])


def correct_weighted_ce_bias(model, class_weights):
    """Bake the fixed inverse cost prior into the head; no validation-fitted threshold.

    Weighted CE has optimum q(c|x) proportional to w[c]*p(c|x). Subtracting
    log(w) undoes that known training bias. Dice means this is approximate;
    the complete validation gate still has to pass after the correction.
    """
    import torch
    weights = torch.as_tensor(class_weights, device=model.head.bias.device, dtype=model.head.bias.dtype)
    if weights.shape != model.head.bias.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Expected one finite positive weight per teacher class')
    with torch.no_grad():
        model.head.bias.sub_(weights.log())


def precision_teacher_loss(logits, target, valid, dice_weight=.25, false_opening_weight=.5):
    """Unweighted likelihood plus explicit wall hard negatives, without prior inflation."""
    import torch
    import torch.nn.functional as F
    logits = logits.float(); valid = valid.float()
    ce = (F.cross_entropy(logits, target, reduction='none') * valid).sum() / valid.sum().clamp_min(1)
    probability = logits.softmax(1)
    truth = F.one_hot(target, 6).permute(0,3,1,2) * valid[:,None]
    masked = probability * valid[:,None]
    dice = 1 - ((2*(masked*truth).sum((0,2,3))+1)
        / (masked.sum((0,2,3))+truth.sum((0,2,3))+1))[1:4].mean()
    wall = (target == 1).float() * valid
    opening_probability = probability[:,2:4].sum(1).clamp(max=1-1e-6)
    hard_negative = (-torch.log1p(-opening_probability)*wall).sum()/wall.sum().clamp_min(1)
    return ce + dice_weight*dice + false_opening_weight*hard_negative


def evaluate_teacher(model, loader, device):
    import torch
    import torch.nn.functional as F
    model.eval(); confusion = torch.zeros(36, dtype=torch.long, device=device)
    by_source = {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        for batch in loader:
            image = F.interpolate(batch['photo'].to(device), (256,256), mode='bilinear', align_corners=False)
            target = F.interpolate(batch['labels'][:,None].float().to(device), (256,256), mode='nearest')[:,0].long()
            valid = F.interpolate(batch['valid'][:,None].float().to(device), (256,256), mode='nearest')[:,0] > 0
            predicted = model(image).argmax(1)
            confusion += torch.bincount((target[valid] * 6 + predicted[valid]), minlength=36)
            for i, sample_id in enumerate(batch.get('id', ['unknown']*len(image))):
                source = sample_id.split('/')[0]
                if source not in by_source: by_source[source] = torch.zeros_like(confusion)
                mask = valid[i]
                by_source[source] += torch.bincount(target[i][mask]*6+predicted[i][mask], minlength=36)
    scores = segmentation_scores(confusion.reshape(6,6).cpu().numpy())
    scores['by_source'] = {k:segmentation_scores(v.reshape(6,6).cpu().numpy()) for k,v in by_source.items()}
    return scores


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file, load_file
    from experiments.facade_semantic_v2.data import validate_dataset
    from experiments.facade_sd15_controlnet.train import git_revision
    if not torch.cuda.is_available(): raise RuntimeError('Teacher training requires Kaggle GPU')
    config = json.loads(Path(args.config).read_text()); torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    validate_dataset(args.data)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    data = MaterialDataset(args.data, 'train', augment=True, crop_probability=.5, blank_probability=.2)
    val = MaterialDataset(args.data, 'val')
    loader_val = DataLoader(val, batch_size=8, num_workers=0)
    device = torch.device('cuda'); model = make_teacher(config['teacher_width']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['teacher_lr'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda'); epochs = 1 if args.smoke else config['teacher_epochs']
    tracker = RunTracker(output, 'ArchiGen-Facade-V3-Semantic-Teacher',
        {'epochs': epochs, 'width': config['teacher_width'], 'resolution':256, 'seed':config['seed'],
         'manifest_sha256':digest(Path(args.data)/'manifest.json'), 'training_samples':len(data),
         'loss':'unweighted CE + soft Dice + explicit wall hard negatives',
         'dice_weight':config['teacher_dice_weight'],'hard_negative_weight':config['teacher_hard_negative_weight'],
         'recovery_source_sha256':config['teacher_recovery_sha256'] if getattr(args,'initialize_from',None) else 'none',
         'acceptance_gate':{k:v for k,v in config.items() if k.startswith(('teacher_min_','teacher_max_'))}},
        {'git_commit':git_revision(), 'mode':'smoke' if args.smoke else 'train', 'test_used':False},
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    material_audit=Path(args.data)/'material-reference-audit.json'
    if material_audit.is_file():tracker.artifact(material_audit)
    best = (-1, -float('inf')); step = 0; source_metadata = None
    def save_candidate(scores, extra_epochs):
        nonlocal best
        key = teacher_selection_key(scores, config)
        if key <= best: return
        best = key
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(output/'teacher.safetensors'))
        metadata={'architecture':'facade_unet_gn_v3','width':config['teacher_width'],'classes':6,'input_range':[-1,1],
            'resolution':256,'trained_epochs':(source_metadata or {}).get('trained_epochs',0)+extra_epochs,
            'trained_steps':(source_metadata or {}).get('trained_steps',0)+step,'additional_optimizer_steps':step,
            'smoke_only':args.smoke,'manifest_sha256':digest(Path(args.data)/'manifest.json'),
            'validation':scores,'run_id':tracker.run_id,'weights_sha256':digest(output/'teacher.safetensors'),
            'test_used':False,'selection_policy':'acceptance gate first, then overlap minus wall false-positive rate',
            'feedback_accepted':bool(teacher_accepted(scores,config) and not args.smoke),
            'recovery':None if source_metadata is None else {
                'source_weights_sha256':source_metadata['weights_sha256'], 'source_run_id':source_metadata['run_id'],
                'fixed_inverse_ce_weights':config['teacher_legacy_class_weights'],
                'validation_fitted_calibration':False}}
        write_json(output/'teacher.json',metadata)
    try:
        if getattr(args,'initialize_from',None) and not args.smoke:
            seed_root = Path(args.initialize_from)
            source_metadata = json.loads((seed_root/'teacher.json').read_text())
            if source_metadata['weights_sha256'] != config['teacher_recovery_sha256'] or digest(seed_root/'teacher.safetensors') != config['teacher_recovery_sha256']:
                raise ValueError('Recovery teacher checksum mismatch')
            if source_metadata['manifest_sha256'] != digest(Path(args.data)/'manifest.json') or source_metadata['smoke_only']:
                raise ValueError('Recovery teacher split mismatch or smoke weights')
            model.load_state_dict(load_file(str(seed_root/'teacher.safetensors')))
            correct_weighted_ce_bias(model,config['teacher_legacy_class_weights'])
            scores=evaluate_teacher(model,loader_val,device);save_candidate(scores,0)
            tracker.metrics({k:v for k,v in scores.items() if isinstance(v,(int,float))},0)
            write_json(output/'recovery-validation.json',{'scores':scores,'passed':teacher_accepted(scores,config),
                'correction':'subtract log of original CE class weights; fixed without fitting validation labels',
                'test_used':False,'validation_is_independent_test':False})
            tracker.artifact(output/'recovery-validation.json')
            print('TEACHER RECOVERY VALIDATION: '+json.dumps(scores),flush=True)
            if teacher_accepted(scores,config):
                tracker.artifact(output/'teacher.json','model');tracker.finish()
                print('TEACHER GATE PASSED: all original thresholds retained; starting diffusion next',flush=True)
                return
        for epoch in range(epochs):
            data.epoch=epoch; model.train(); losses=[]
            loader = DataLoader(data,batch_size=config['teacher_batch_size'],
                sampler=data.training_order(epoch),num_workers=0)
            for batch in loader:
                image=F.interpolate(batch['photo'].to(device),(256,256),mode='bilinear',align_corners=False)
                target=F.interpolate(batch['labels'][:,None].float().to(device),(256,256),mode='nearest')[:,0].long()
                valid=F.interpolate(batch['valid'][:,None].float().to(device),(256,256),mode='nearest')[:,0]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(image)
                    loss=precision_teacher_loss(logits,target,valid,
                        config['teacher_dice_weight'],config['teacher_hard_negative_weight'])
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update();step+=1;losses.append(float(loss.detach()))
                if args.smoke and step>=2:break
            scores=evaluate_teacher(model,loader_val if not args.smoke else [next(iter(loader_val))],device)
            metrics={k:v for k,v in scores.items() if isinstance(v,(int,float))}
            tracker.metrics({'train_loss':float(np.mean(losses)),**metrics},epoch+1)
            print(json.dumps({'teacher_epoch':epoch+1,'optimizer_steps':step,**metrics}),flush=True)
            save_candidate(scores,epoch+1)
        metadata=json.loads((output/'teacher.json').read_text()); scores=metadata['validation']
        accepted=teacher_accepted(scores,config)
        metadata['feedback_accepted']=bool(accepted and not args.smoke);write_json(output/'teacher.json',metadata)
        tracker.artifact(output/'teacher.json','model')
        if not args.smoke and not accepted:
            raise RuntimeError('Teacher failed the predeclared validation gate; diffusion training is blocked')
        tracker.finish()
    except BaseException:
        tracker.finish('FAILED');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.json')));parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--initialize-from',help='Checksummed previous teacher; fixed inverse class-cost correction, then validate before any further fitting')
    train(parser.parse_args())
