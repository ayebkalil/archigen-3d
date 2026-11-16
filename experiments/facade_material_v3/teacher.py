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


def evaluate_teacher(model, loader, device):
    import torch
    import torch.nn.functional as F
    model.eval(); confusion = torch.zeros(36, dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        for batch in loader:
            image = F.interpolate(batch['photo'].to(device), (256,256), mode='bilinear', align_corners=False)
            target = F.interpolate(batch['labels'][:,None].float().to(device), (256,256), mode='nearest')[:,0].long()
            valid = F.interpolate(batch['valid'][:,None].float().to(device), (256,256), mode='nearest')[:,0] > 0
            predicted = model(image).argmax(1)
            confusion += torch.bincount((target[valid] * 6 + predicted[valid]), minlength=36)
    return segmentation_scores(confusion.reshape(6,6).cpu().numpy())


def train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from safetensors.torch import save_file
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
         'manifest_sha256':digest(Path(args.data)/'manifest.json'), 'training_samples':len(data)},
        {'git_commit':git_revision(), 'mode':'smoke' if args.smoke else 'train', 'test_used':False},
        tracking_uri=os.getenv('MLFLOW_TRACKING_URI'))
    material_audit=Path(args.data)/'material-reference-audit.json'
    if material_audit.is_file():tracker.artifact(material_audit)
    best = -1.; step = 0
    try:
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
                    ce=F.cross_entropy(logits.float(),target,reduction='none',weight=logits.new_tensor([.2,1,3,4,1,1]).float())
                    probabilities=logits.float().softmax(1)*valid[:,None]
                    truth=F.one_hot(target,6).permute(0,3,1,2)*valid[:,None]
                    dice=1-((2*(probabilities*truth).sum((0,2,3))+1)/(probabilities.sum((0,2,3))+truth.sum((0,2,3))+1))[1:4].mean()
                    loss=(ce*valid).sum()/valid.sum().clamp_min(1)+dice
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update();step+=1;losses.append(float(loss.detach()))
                if args.smoke and step>=2:break
            scores=evaluate_teacher(model,loader_val if not args.smoke else [next(iter(loader_val))],device)
            selection=.5*scores['window_iou']+.3*scores['door_iou']+.2*scores['wall_iou']
            metrics={k:v for k,v in scores.items() if isinstance(v,(int,float))}
            tracker.metrics({'train_loss':float(np.mean(losses)),**metrics},epoch+1)
            print(json.dumps({'teacher_epoch':epoch+1,'optimizer_steps':step,**metrics}),flush=True)
            if selection>best:
                best=selection
                save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(output/'teacher.safetensors'))
                metadata={'architecture':'facade_unet_gn_v3','width':config['teacher_width'],'classes':6,'input_range':[-1,1],
                    'resolution':256,'trained_epochs':epoch+1,'trained_steps':step,'smoke_only':args.smoke,
                    'manifest_sha256':digest(Path(args.data)/'manifest.json'),'validation':scores,'run_id':tracker.run_id,
                    'weights_sha256':digest(output/'teacher.safetensors'),'test_used':False}
                write_json(output/'teacher.json',metadata)
        metadata=json.loads((output/'teacher.json').read_text()); scores=metadata['validation']
        accepted=(scores['miou']>=config['teacher_min_miou'] and scores['opening_iou']>=config['teacher_min_opening_iou']
            and scores['door_iou']>=config['teacher_min_door_iou'] and scores['wall_false_opening_rate']<=config['teacher_max_wall_fp'])
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
    train(parser.parse_args())
