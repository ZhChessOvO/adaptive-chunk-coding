"""Fixed native-pixel panels for the three previously recorded text examples.

Purely display existing teacher outputs. No resizing, sharpening, generative
editing, OCR relabelling, ground-truth approval, model inference or training.
"""
import math
import os
from pathlib import Path

from PIL import Image,ImageDraw
from demo.routervc_content_review import ROOT,paired_pixels
from demo.routervc_fullview_probe import read,digest,save,verify_artifacts


def crop_rect(box,shape,padding=8):
    x,y,w,h=box;height,width=shape[:2]
    return [max(0,math.floor(x)-padding),max(0,math.floor(y)-padding),
            min(width,math.ceil(x+w)+padding),min(height,math.ceil(y+h)+padding)]


def main():
    if not os.environ.get('TMUX'):raise RuntimeError('use tmux for teacher-pixel reading')
    root=ROOT/'content_review_visuals';root.mkdir(exist_ok=True)
    reference=ROOT/'content_source_visual_review.json';review=read(reference)
    rows={r['sample_id']:r for r in read(ROOT/'mixedview_teacher/labels.json')['samples']}
    explicit={'uvg-jockey-f032-center':1,'reds-train-158-f000-n17-fullview':11,
              'reds-train-236-f000-n17-fullview':3}
    records=[]
    for item in review['reviewed_items']:
        sid=item['sample_id'];region=explicit[sid];row=rows[sid]
        if digest(row['path'])!=row['sha256']:raise ValueError('teacher record changed')
        source,candidates,roi,pixels=paired_pixels(read(row['path']),region)
        frame=item['frame_index_within_17']
        rect=crop_rect(item['box_xywh'],source.shape[1:]);left,top,right,bottom=rect
        width,height=right-left,bottom-top
        canvas=Image.new('RGB',(width*5,height+34),'white');draw=ImageDraw.Draw(canvas)
        for i,(title,video) in enumerate([('Source',source),*candidates.items()]):
            draw.text((width*i+5,9),title,fill='black')
            canvas.paste(Image.fromarray(video[frame,top:bottom,left:right]),(width*i,34))
        target=root/(sid+'.png');temporary=target.with_suffix('.tmp')
        canvas.save(temporary,format='PNG');os.replace(temporary,target)
        records.append(dict(sample_id=sid,region=region,frame=frame,crop_xyxy=rect,
            previous_source_review=digest(reference),source_text_reference=item['source_reference_text'],
            source_reference_is_human_annotation=False,candidates_verified_for_training=False,
            pixel_binding=pixels,path=target.name,sha256=digest(target),resized=False))
    save(root/'complete.json',dict(complete=True,records=records,
        artifacts={r['path']:r['sha256'] for r in records},new_metrics=False,new_inference=False))
    print('CONTENT_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':main()
