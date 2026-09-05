#!/usr/bin/env python3
"""build_3x_full.py — 全量 3x 合成图标注构建 (24678 图)
========================================================================
3x 合成图 = 真实源图目标放大 3 倍: {stem}_rank{1..3}.png
bbox = 源标注 × 缩放 (sx=w/src_w, sy=h/src_h), 与 build_train_ship3x_full.py 同机制

- 源: finaldatav5/annotations_train.json (12049 真实图)
- 合成池: airplane_3x (20472 图, 飞机源) + ship_3x (2805, 舰船源) + fsc_3x (1455, 发射车源)
- 命中: airplane 20472/20472, ship 2805/2805, fsc 1401/1455 (54 张无源跳过)
- 输出: finaldatav5/annotations_train_3x_full.json (全量合成池, 绝对路径)
"""
import json
from collections import Counter
from pathlib import Path
from PIL import Image

SRC_ANN = '/home/jingyue/datasets/finaldatav5/annotations_train.json'
OUT_ANN = '/home/jingyue/datasets/finaldatav5/annotations_train_3x_full.json'
DIRS = [
    ('/home/jingyue/finaldatav2_airplane_3x_output/final', 20472),
    ('/home/jingyue/finaldatav2_ship_3x_output/final', 2805),
    ('/home/jingyue/finaldatav2_fsc_3x_output/final', 1455),
]

coco = json.load(open(SRC_ANN))
imgid2info = {im['id']: im for im in coco['images']}
stem2info = {Path(im['file_name']).stem: im for im in coco['images']}
bbox_by_stem = {}
for a in coco['annotations']:
    stem = Path(imgid2info[a['image_id']]['file_name']).stem
    bbox_by_stem.setdefault(stem, []).append(a)

new_images, new_anns = [], []
skip_no_stem = 0
for final_dir, n_total in DIRS:
    gen_files = sorted(Path(final_dir).glob('*.png'))
    print(f'{final_dir.split("/")[-1]}: {len(gen_files)} 图 (期望 {n_total})')
    for fp in gen_files:
        base = fp.name[:-4]
        stem = base.rsplit('_rank', 1)[0] if '_rank' in base else base
        if stem not in stem2info:
            skip_no_stem += 1
            print(f'  WARN no source: {fp.name}')
            continue
        src = stem2info[stem]
        anns = bbox_by_stem.get(stem, [])
        if not anns:
            continue
        with Image.open(fp) as im:
            w, h = im.size
        sx, sy = w / src['width'], h / src['height']
        img_id = len(new_images)
        new_images.append({'id': img_id, 'file_name': str(fp),
                           'width': w, 'height': h})
        for a in anns:
            x, y, bw, bh = a['bbox']
            x1, y1 = x * sx, y * sy
            x2, y2 = (x + bw) * sx, (y + bh) * sy
            new_anns.append({
                'id': len(new_anns), 'image_id': img_id,
                'category_id': a['category_id'],
                'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                'area': round((x2 - x1) * (y2 - y1), 2), 'iscrowd': 0,
            })

out = {'images': new_images, 'annotations': new_anns,
       'categories': coco['categories']}
json.dump(out, open(OUT_ANN, 'w'), ensure_ascii=False)
cnt = Counter(a['category_id'] for a in new_anns)
groups = {'舰船(0-3)': sum(v for k, v in cnt.items() if k < 4),
          '飞机(4-23)': sum(v for k, v in cnt.items() if 4 <= k < 24),
          '发射车(24)': cnt.get(24, 0)}
print(f'OUT: {OUT_ANN}')
print(f'总图: {len(new_images)} (跳过无源 {skip_no_stem}) / 标注: {len(new_anns)}')
print('类别分布:', groups)
