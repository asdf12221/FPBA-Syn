#!/usr/bin/env python3
"""预训练数据: 纯合成 bal300 取 2/3 (按图主要类别分层抽样, 保持类别平衡)

- synth:  annotations_train_synth_bal300.json 4572 图
  = airplane_3x (3745) + ship_3x (501) + fsc_3x (326) 三目录的平衡抽样版
- 分层:   按每图标注最多的细类分组, 每组抽 2/3 (seed=2026 固定可复现)
- 纯合成, 不含任何真实数据; categories 用 synth 的 (25 类)
- file_name 已是绝对路径 → data_root='/' 加载
"""
import json
import random
from collections import Counter

SYN = '/home/jingyue/datasets/finaldatav5/annotations_train_synth_bal300.json'
OUT = '/home/jingyue/datasets/finaldatav5/annotations_train_synth_2of3.json'

syn = json.load(open(SYN))

# 每图主要类别
im_main = {}
for a in syn['annotations']:
    im_main.setdefault(a['image_id'], Counter())[a['category_id']] += 1
main_cat = {im['id']: im_main[im['id']].most_common(1)[0][0]
            for im in syn['images'] if im['id'] in im_main}

# 分层抽样: 每类取 2/3
random.seed(2026)
by_cat = {}
for im in syn['images']:
    c = main_cat.get(im['id'], -1)
    by_cat.setdefault(c, []).append(im)
sel_ids = set()
for c, imgs in sorted(by_cat.items()):
    k = int(round(len(imgs) * 2 / 3))
    chosen = random.sample(imgs, k)
    sel_ids.update(im['id'] for im in chosen)
    print(f'cat {c}: {len(imgs)} → {k}')

imgs = []
for im in syn['images']:
    if im['id'] in sel_ids:
        im2 = dict(im)
        im2['id'] = len(imgs)
        imgs.append(im2)
id_map = {old: new for new, old in enumerate(sorted(sel_ids))}

anns = []
for aid, a in enumerate(syn['annotations']):
    if a['image_id'] in id_map:
        a2 = dict(a)
        a2['id'] = aid
        a2['image_id'] = id_map[a['image_id']]
        anns.append(a2)

out = {'images': imgs, 'annotations': anns, 'categories': syn['categories']}
json.dump(out, open(OUT, 'w'), ensure_ascii=False)
print(f'TOTAL imgs={len(imgs)} anns={len(anns)}')
print('OUT:', OUT)
