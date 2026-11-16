"""Guard pilot split selection and shared native model input geometry."""
import json

import numpy as np
from PIL import Image

from experiments.facade_material_v3.compare_current import prepare


def test_matched_pilot_uses_only_validation_and_preserves_native_encodings(tmp_path):
    source=tmp_path/'cmp';source.mkdir();project=tmp_path/'project'
    bank=project/'frontend/public/showcase/tex-1k/white_stucco';bank.mkdir(parents=True)
    Image.new('RGB',(32,32),'white').save(bank/'diff.jpg')
    rows=[]
    for i in range(2):
        labels=np.ones((80,160),np.uint8);labels[20:60,10:150]=2;labels[30:50,50:70]=3
        photo=np.full((80,160,3),[140,100+i*20,80],np.uint8)
        Image.fromarray(labels).save(source/f'{i}-labels.png');Image.fromarray(photo).save(source/f'{i}.png')
        rows.append({'id':f'val/{i}','split':'val','photo':f'{i}.png','labels':f'{i}-labels.png'})
    rows.extend([{'id':'unread-test','split':'test','photo':'missing.jpg','labels':'missing.png'},
        {'id':'unread-train','split':'train','photo':'missing2.jpg','labels':'missing2.png'}])
    (source/'manifest.json').write_text(json.dumps({'samples':rows}))
    output=tmp_path/'comparison';prepare(project,source,output,count=2)
    cases=json.loads((output/'cases.json').read_text())
    assert {r['id'] for r in cases}=={'val/0','val/1'}
    assert not json.loads((output/'protocol.json').read_text())['test_used']
    for case in cases:
        folder=output/case['folder']
        assert Image.open(folder/'production-input.png').size==(256,256)
        assert Image.open(folder/'target.png').size==(512,512)
        valid=np.asarray(Image.open(folder/'valid.png'))>0
        y,x=np.where(valid)
        assert abs((x.max()-x.min()+1)/(y.max()-y.min()+1)-3.5)<.03
        assert (np.asarray(Image.open(folder/'target.png'))[~valid]==127).all()
        labels=np.asarray(Image.open(folder/'labels.png'))
        assert set(np.unique(labels))=={0,1,2}
    again=tmp_path/'again';prepare(project,source,again,count=2)
    assert cases==json.loads((again/'cases.json').read_text())
