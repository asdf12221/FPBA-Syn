#!/usr/bin/env python3
"""全量合成预训练数据: 三个 3x 目录全部图 + v5 共享标签 → 按细类平衡抽 2/3

- 合成图: {stem}_rank{r}.png, bbox 与 v5 共享 (stem 在 annotations_train.json 里)
- bbox 变换: 源图 W×H → 合成图 nw×nh, 等比缩放 (sx=sy=nw/W)
- 抽样:   按每图主要细类分层, 每类取 2/3 (seed=2026 固定), 纯合成
- 输出:   file_name 绝对路径 → data_root='/' 加载
"""
import json
import random
import os
import re
from collections import Counter, defaultdict
from PIL import Image

V5 = '/home/jingyue/datasets/finaldatav5/annotations_train.json'
OUT = '/home/jingyue/datasets/finaldatav5/annotations_train_synth_full_2of3.json'
DIRS = [
    '/home/jingyue/finaldatav2_airplane_3x_output/final',
    '/home/jingyue/finaldatav2_fsc_3x_output/final',
    '/home/jingyue/finaldatav2_ship_3x_output/final',
]

v5 = json.load(open(V5))
# stem → 图记录 + 标注
stem_img = {}
stem_anns = defaultdict(list)
for im in v5['images']:
    stem = os.path.splitext(im['file_name'].split('/')[-1])[0]
    stem_img[stem] = im
for a in v5['annotations']:
    if a['image_id'] in stem_img:
        stem_anns[a['image_id']].append(a)

total, mapped, missing = 0, 0, 0
miss_stems = Counter()
imgs, anns = [], []
aid = 0
for d in DIRS:
    files = sorted(f for f in os.listdir(d) if f.endswith('.png'))
    for fn in files:
        total += 1
        stem = re.sub(r'_rank\d+$', '', os.path.splitext(fn)[0])
        src = stem_img.get(stem)
        if src is None:
            missing += 1
            miss_stems[stem.split('_')[0]] += 1
            continue
        # 合成图实际尺寸
        with Image.open(os.path.join(d, fn)) as im:
            nw, nh = im.size
        sw, sh = nw / src['width'], nh / src['height']
        im_id = len(imgs)
        imgs.append({
            'id': im_id, 'file_name': os.path.join(d, fn),
            'width': nw, 'height': nh,
        })
        for a in stem_anns[src['id']]:
            bx, by, bw, bh = a['bbox']
            anns.append({
                'id': aid, 'image_id': im_id, 'category_id': a['category_id'],
                'bbox': [bx * sw, by * sh, bw * sw, bh * sh],
                'area': a['area'] * sw * sh, 'iscrowd': a.get('iscrowd', 0),
            })
            aid += 1
        mapped += 1
print(f'合成图总数 {total}: 映射 {mapped}, 未找到源标注 {missing}'
      f'({dict(miss_stems.most_common(5))})')

# --- 按细类分层抽 2/3 ---
im_main = defaultdict(Counter)
for a in anns:
    im_main[a['image_id']][a['category_id']] += 1
main_cat = {im['id']: im_main[im['id']].most_common(1)[0][0] for im in imgs}
random.seed(2026)
by_cat = defaultdict(list)
for im in imgs:
    by_cat[main_cat[im['id']]].append(im['id'])
sel = set()
for c, ids in sorted(by_cat.items()):
    k = int(round(len(ids) * 2 / 3))
    sel.update(random.sample(ids, k))

imgs2 = []
for im in imgs:
    if im['id'] in sel:
        im2 = dict(im)
        im2['id'] = len(imgs2)
        imgs2.append(im2)
id_map = {old: new for new, old in enumerate(sorted(sel))}
anns2 = []
for a in anns:
    if a['image_id'] in id_map:
        a2 = dict(a)
        a2['id'] = len(anns2)
        a2['image_id'] = id_map[a['image_id']]
        anns2.append(a2)

out = {'images': imgs2, 'annotations': anns2, 'categories': v5['categories']}
json.dump(out, open(OUT, 'w'), ensure_ascii=False)
g = defaultdict(int)
for a in anns2:
    g['舰船' if a['category_id'] < 4 else ('飞机' if a['category_id'] < 24 else '发射车')] += 1
print(f'TOTAL imgs={len(imgs2)} anns={len(anns2)}  标注分布: {dict(g)}')
print('OUT:', OUT)
