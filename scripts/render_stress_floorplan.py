"""Plot final vectorized rooms over a canvas reconstructed from the recorded input.

This is an offline rendering of the JSON output, not a new model inference.
"""
import argparse
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();directory=args.directory
    record=json.loads((directory/'record.json').read_text(encoding='utf-8'))
    output=json.loads((directory/'response.json').read_text(encoding='utf-8'))
    image=Image.open(io.BytesIO((directory/'input.bin').read_bytes())).convert('RGB')
    scale=min(512/image.width,512/image.height)
    resized=image.resize((int(image.width*scale),int(image.height*scale)),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(512,512),'white')
    canvas.paste(resized,((512-resized.width)//2,(512-resized.height)//2))
    canvas.save(directory/'reconstructed-floorplan-canvas.png')
    overlay=canvas.convert('RGBA');layer=Image.new('RGBA',canvas.size);draw=ImageDraw.Draw(layer)
    for index,room in enumerate(output.get('rooms',[])):
        polygon=[(x/.03,y/.03) for x,y in room['polygon_points']]
        color=((index*73+60)%256,(index*137+80)%256,(index*199+100)%256,65)
        draw.polygon(polygon,fill=color,outline=(*color[:3],255),width=2)
        x=sum(p[0] for p in polygon)/len(polygon);y=sum(p[1] for p in polygon)/len(polygon)
        draw.text((x,y),str(index+1),fill='red')
    Image.alpha_composite(overlay,layer).convert('RGB').save(directory/'floorplan-output.png')
    (directory/'vector-visualization.json').write_text(json.dumps({
        'scope':'offline final vectorized room polygons; reconstructed letterbox canvas',
        'scale_m_per_canvas_pixel':.03,'input_sha256':record['parameters']['input_sha256'],
        'note':'Raw neural segmentation overlays remain separate. Earlier model-input.png may be the last facade tensor; use the explicitly named reconstructed canvas.'},indent=2))


if __name__=='__main__':main()
