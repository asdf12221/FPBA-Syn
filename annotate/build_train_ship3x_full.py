#!/usr/bin/env python3
"""全量合并 ship Stage3 生成图 (2805 张) 到 finaldatav5 train:
- 源标注: finaldatav5/annotations_train.json 中 ship 类 (0-3) bbox, 按生成图实际尺寸缩放 (sx=w/src_w, sy=h/src_h)
- 输出: annotations_train_ship3x_full.json (原 12049 图 + 2805 生成图 = 14854 图)
- 生成图 symlink 到 images/all/ (缺失才建)
"""
import json, os
from pathlib import Path
from PIL import Image

SRC_ANN = '/home/jingyue/datasets/finaldatav5/annotations_train.json'
GEN_DIR = Path('/home/jingyue/finaldatav2_ship_3x_output/final')
OUT_ANN = '/home/jingyue/datasets/finaldatav5/annotations_train_ship3x_full.json'
LINK_DIR = Path('/home/jingyue/datasets/finaldatav5/images/all')
SHIP_IDS = set(range(0, 4))

coco = json.load(open(SRC_ANN))
imgid2info = {im['id']: im for im in coco['images']}
stem2info = {Path(im['file_name']).stem: im for im in coco['images']}
bbox_by_stem = {}
for a in coco['annotations']:
    if a['category_id'] in SHIP_IDS:
        stem = Path(imgid2info[a['image_id']]['file_name']).stem
        bbox_by_stem.setdefault(stem, []).append(a)

gen_files = sorted(GEN_DIR.glob('*.png'))
print(f'generated images: {len(gen_files)}')

new_images, new_anns = [], []
n_skip_nostem = n_skip_noann = 0
next_img_id = max(imgid2info) + 1
next_ann_id = max(a['id'] for a in coco['annotations']) + 1

for fp in gen_files:
    name = fp.name
    # <stem>_rank{1..3}.png
    base = name[:-len('.png')]
    assert base.endswith('_rank') or '_rank' in base
    stem = base.rsplit('_rank', 1)[0]
    if stem not in stem2info:
        n_skip_nostem += 1
        print(f'WARN no source image for: {name}')
        continue
    src = stem2info[stem]
    anns = bbox_by_stem.get(stem, [])
    if not anns:
        n_skip_noann += 1
        print(f'WARN no ship ann for: {name}')
        continue
    with Image.open(fp) as im:
        w, h = im.size
    sx, sy = w / src['width'], h / src['height']
    new_images.append({'id': next_img_id, 'file_name': name, 'width': w, 'height': h})
    for a in anns:
        x, y, bw, bh = a['bbox']
        x2, y2 = (x + bw) * sx, (y + bh) * sy
        x1, y1 = x * sx, y * sy
        new_anns.append({
            'id': next_ann_id, 'image_id': next_img_id, 'category_id': a['category_id'],
            'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            'area': round((x2 - x1) * (y2 - y1), 2), 'iscrowd': 0,
        })
        next_ann_id += 1
    next_img_id += 1
    # symlink
    link = LINK_DIR / name
    if not link.exists():
        os.symlink(fp, link)

print(f'new images: {len(new_images)}, new anns: {len(new_anns)}')
print(f'skip (no source): {n_skip_nostem}, skip (no ann): {n_skip_noann}')

out = dict(coco)
out['images'] = coco['images'] + new_images
out['annotations'] = coco['annotations'] + new_anns
json.dump(out, open(OUT_ANN, 'w'))
print(f'written: {OUT_ANN}')
print(f'total images: {len(out["images"])}, total anns: {len(out["annotations"])}')

# 类别分布 (新增)
from collections import Counter
cnt = Counter(a['category_id'] for a in new_anns)
print('new ann cat dist:', dict(sorted(cnt.items())))
