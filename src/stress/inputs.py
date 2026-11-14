"""Deterministic input generation, always invoked inside a bounded child process."""
import argparse
import io
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def plan_geometry(count):
    columns = math.ceil(math.sqrt(count)); rows = math.ceil(count/columns)
    rooms, doors, windows = [], [], []
    for i in range(count):
        x,y = i%columns,i//columns
        x0,y0 = 16+x*992/columns,16+y*992/rows
        x1,y1 = 16+(x+1)*992/columns,16+(y+1)*992/rows
        rooms.append([[x0+2,y0+2],[x1-2,y0+2],[x1-2,y1-2],[x0+2,y1-2]])
        doors.append([[x0+8,y0],[x0+16,y0]])
        windows.append([[x0+25,y0],[x0+40,y0]])
    return {'source_dimensions':[1024,1024],'canvas_dimensions':[512,512],
            'scale_m_per_canvas_pixel':.03,'room_polygons_source_px':rooms,
            'door_segments_source_px':doors,'window_segments_source_px':windows,
            'note':'Synthetic geometry only; no room-type semantic annotation'}


def cases(config, model='furniture'):
    result = []
    def add(family, transform, value=None, **kw):
        suffix = '-'+kw['format'] if 'format' in kw else ''
        result.append(dict(id=(f'{transform}-{value}' if value is not None else transform)+suffix,
                           family=family, transform=transform, value=value, **kw))
    for size in config['source_sizes']:
        for fmt in ('png', 'jpeg'):
            add('resolution', 'resize', size, format=fmt)
    for transform in ('blur', 'noise', 'contrast', 'jpeg', 'mask', 'rotate'):
        for value in config[transform]:
            add('degradation', transform, value)
    if model == 'furniture':
        for transform in ('complex_background', 'multiple_objects', 'incomplete', 'unusual_category'):
            add('furniture', transform)
    elif model == 'floorplan':
        for rooms in config['plan_rooms']: add('plan_complexity','plan_grid',rooms)
        for path in sorted(Path(config.get('plan_directory','data/external_test_plans')).glob('*.png')):
            add('unusual', 'external_plan', path.name, source=str(path))
    else:
        # API accepts architectural wall geometry, not these image transformations.
        result = []
        for floors in (1, 4, 16, 64):
            for openings in (2, 16, 64):
                add('facade_api','wall_geometry',f'{floors}-{openings}', payload={
                    'style':'Modern Minimalist','num_floors':floors,
                    'wall':{'id':'stress_wall','is_exterior':True,
                            'polygon_meters':[[0,0],[32,0],[32,.25],[0,.25]],
                            'openings':[{'type':'window','position_along_wall_m':(i+.5)*32/openings,
                                         'width':.6,'height':1.2} for i in range(openings)]}})
        add('facade_api','unknown_style', payload={'style':'Unknown stress style','num_floors':1})
        for size in config['source_sizes']:
            add('facade_semantic','resize',size,format='png',semantic=True)
        for transform in ('blur','noise','contrast','jpeg','mask','rotate'):
            for value in config[transform]: add('facade_semantic',transform,value,semantic=True)
    for transform in ('empty', 'uniform', 'transparent', 'truncated', 'unsupported', 'zero_dimension', 'extreme_header', 'extreme_aspect'):
        add('invalid', transform, expected_rejection=transform not in ('uniform', 'extreme_aspect'), **({'semantic':True} if model=='facade' else {}))
    priority = {'resolution':0,'plan_complexity':1,'unusual':2,'facade_api':0,'facade_semantic':1,
                'degradation':3,'furniture':4,'invalid':5}
    return sorted(result,key=lambda c:priority.get(c['family'],6))


def create(case, source, seed, alpha_output=None):
    rng = np.random.default_rng(seed)
    t, value = case['transform'], case.get('value')
    if t == 'empty': return b''
    if t == 'unsupported': return b'This is a text file, not an image.'
    if t in ('zero_dimension', 'extreme_header'):
        w, h = (0, 512) if t == 'zero_dimension' else (1000000, 1000000)
        def chunk(kind, payload):
            return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
        return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(b'')) + chunk(b'IEND', b'')
    image = Image.open(source).convert('RGBA')
    known_alpha = None
    if case.get('source'):
        image = Image.open(case['source']).convert('RGBA')
    if case.get('semantic'):
        image = Image.new('RGBA',(256,256),(0,0,255,255))
        draw = ImageDraw.Draw(image)
        for x in (24,100,176): draw.rectangle((x,40,x+40,100),fill=(255,255,0,255))
    if t == 'plan_grid':
        image = Image.new('RGBA',(1024,1024),'white')
        draw = ImageDraw.Draw(image)
        columns = math.ceil(math.sqrt(value)); rows = math.ceil(value/columns)
        for i in range(value):
            x,y = i%columns, i//columns
            x0,y0 = 16+x*992/columns,16+y*992/rows
            x1,y1 = 16+(x+1)*992/columns,16+(y+1)*992/rows
            draw.rectangle((x0,y0,x1,y1),outline='black',width=4)
            draw.line((x0+8,y0,x0+16,y0),fill='white',width=5)
            draw.line((x0+8,y0,x0+16,y0+8),fill='black',width=1)
            draw.line((x0+25,y0,x0+40,y0),fill='white',width=5)
            draw.line((x0+25,y0-1,x0+40,y0-1),fill='black',width=1)
            draw.line((x0+25,y0+1,x0+40,y0+1),fill='black',width=1)
    elif t == 'resize':
        image.thumbnail((value, value)) if max(image.size) > value else None
        ratio = value / max(image.size)
        image = image.resize((max(1, round(image.width*ratio)), max(1, round(image.height*ratio))))
    elif t == 'blur': image = image.filter(ImageFilter.GaussianBlur(value))
    elif t == 'noise':
        a = np.asarray(image).copy()
        a[..., :3] = np.clip(a[..., :3].astype(float) + rng.normal(0, value, a[..., :3].shape), 0, 255)
        image = Image.fromarray(a)
    elif t == 'contrast': image = ImageEnhance.Contrast(image).enhance(value)
    elif t == 'mask':
        ImageDraw.Draw(image).rectangle((image.width*(1-value), 0, image.width, image.height), fill=(127,127,127,255))
    elif t == 'rotate': image = image.rotate(value, expand=True)
    elif t == 'complex_background':
        known_alpha = image.getchannel('A')
        background = Image.fromarray(rng.integers(0, 256, (*np.asarray(image).shape[:2], 3), dtype=np.uint8)).convert('RGBA')
        background.alpha_composite(image)
        image = background
    elif t == 'multiple_objects':
        background = Image.new('RGBA', (image.width*2, image.height))
        background.alpha_composite(image, (0,0)); background.alpha_composite(image, (image.width,0))
        image = background
    elif t == 'incomplete': image = image.crop((0, 0, image.width, int(image.height*.55)))
    elif t == 'uniform': image = Image.new('RGB', (512,512), 'white')
    elif t == 'transparent': image = Image.new('RGBA', (512,512), (0,0,0,0))
    elif t == 'extreme_aspect': image = image.resize((16384,1))
    fmt = 'JPEG' if case.get('format') == 'jpeg' or t == 'jpeg' else 'PNG'
    if alpha_output is not None and (fmt=='JPEG' or t=='complex_background'):
        (known_alpha or image.getchannel('A')).save(alpha_output)
    if fmt == 'JPEG':
        background = Image.new('RGB', image.size, (127,127,127))
        background.paste(image, mask=image.getchannel('A') if image.mode == 'RGBA' else None)
        image = background
    out = io.BytesIO()
    image.save(out, format=fmt, **({'quality': value if t == 'jpeg' else 90} if fmt == 'JPEG' else {}))
    data = out.getvalue()
    return data[:len(data)//3] if t == 'truncated' else data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    case = json.loads(args.case)
    data = create(case, args.source, args.seed, Path(args.output).with_name('expected-alpha.png'))
    Path(args.output).write_bytes(data)
    if case['transform']=='plan_grid':
        Path(args.output).with_name('expected-geometry.json').write_text(json.dumps(plan_geometry(case['value']),indent=2))


if __name__ == '__main__': main()
