#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成端侧嵌入词表 assets/prompt_emb.bin
=====================================

App 内置了离线编码器，不依赖任何服务即可运行，但那个编码器<b>不是</b> CLIP 对齐
的。若希望手机上真正用 CLIP 语义做开放词汇检测（且不想让手机连 PC 服务），标准做法
是：**把固定词表的嵌入预先算好、打进 APK**。本脚本就是这个"固化"步骤。

三种来源：
    # 1) 从文本塔服务取（推荐：服务端可用 --clip 跑真实 CLIP 文本塔）
    python3 tools/yoloworld_text_tower.py --clip          # 另开一个终端
    python3 tools/build_prompt_table.py --prompts "人,汽车,自行车,狗" \\
            --server http://127.0.0.1:8000

    # 2) 本地 mock（与服务端 mock 完全一致的确定性向量，仅用于验证链路）
    python3 tools/build_prompt_table.py --prompts "人,汽车,自行车,狗" --mock

    # 3) 从文件读词表（每行一个类别）
    python3 tools/build_prompt_table.py --file prompts.txt --server http://127.0.0.1:8000

生成后重新编译 APK，App 启动时会自动加载：
    app/src/main/assets/prompt_emb.bin

文件格式（小端序，与 TextEmbedder.java 一致）：
    char[4] 'Y','W','E','M'
    uint32  version = 1
    uint32  count
    uint32  dim
    repeat count:
      uint32  nameLen
      byte[nameLen] name (UTF-8)
      float32[dim]  embedding (L2 归一化)
"""
import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import urllib.request

MAGIC = b"YWEM"
VERSION = 1
DEFAULT_DIM = 512


def mock_embedding(text, dim=DEFAULT_DIM):
    """与服务端 --mock 完全一致：sha256(text) 播种 -> 高斯白噪声 -> L2 归一化。"""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def fetch_from_server(server, prompts, timeout=60):
    if not server.startswith("http://") and not server.startswith("https://"):
        server = "http://" + server
    body = json.dumps({"prompts": prompts}).encode("utf-8")
    req = urllib.request.Request(server.rstrip("/") + "/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode("utf-8"))
    embs = res["embeddings"]
    if len(embs) != len(prompts):
        raise RuntimeError("服务返回条数(%d)与请求(%d)不一致" % (len(embs), len(prompts)))
    return embs, len(embs[0])


def write_table(path, names, embs, dim):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VERSION, len(names), dim))
        for name, vec in zip(names, embs):
            nb = name.encode("utf-8")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<%df" % dim, *vec))
    return os.path.getsize(path)


def read_table(path):
    """回读校验，确保 Java 侧能按同样的格式解析。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise RuntimeError("magic 不匹配")
    version, count, dim = struct.unpack_from("<III", data, 4)
    off = 16
    rows = []
    for _ in range(count):
        (name_len,) = struct.unpack_from("<I", data, off)
        off += 4
        name = data[off:off + name_len].decode("utf-8")
        off += name_len
        vec = struct.unpack_from("<%df" % dim, data, off)
        off += 4 * dim
        rows.append((name, vec))
    if off != len(data):
        raise RuntimeError("尾部有 %d 字节残留" % (len(data) - off))
    return version, dim, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=None, help="逗号分隔的类别列表")
    ap.add_argument("--file", default=None, help="每行一个类别的文本文件")
    ap.add_argument("--server", default=None, help="文本塔服务地址，如 http://127.0.0.1:8000")
    ap.add_argument("--mock", action="store_true", help="本地生成 mock 向量（不用服务）")
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM, help="仅 mock 模式使用")
    ap.add_argument("--out", default="app/src/main/assets/prompt_emb.bin")
    args = ap.parse_args()

    names = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            names += [ln.strip() for ln in f if ln.strip()]
    if args.prompts:
        for p in args.prompts.replace("，", ",").split(","):
            p = p.strip()
            if p:
                names.append(p)

    # 去重且保持顺序
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    names = uniq

    if not names:
        print("错误：没有类别。用 --prompts 或 --file 指定。")
        return 1
    if not args.mock and not args.server:
        print("错误：请指定 --server（真实/服务端嵌入）或 --mock（本地占位向量）。")
        return 1

    if args.mock:
        dim = args.dim
        embs = [mock_embedding(n, dim) for n in names]
        src = "本地 mock（%d 维，非 CLIP 对齐）" % dim
    else:
        embs, dim = fetch_from_server(args.server, names)
        src = "文本塔服务 %s（%d 维）" % (args.server, dim)

    size = write_table(args.out, names, embs, dim)
    version, rdim, rows = read_table(args.out)

    print("来源   : %s" % src)
    print("类别   : %d 个" % len(names))
    print("维度   : %d" % rdim)
    print("写出   : %s (%.1f KB)" % (args.out, size / 1024.0))
    print("回读校验: version=%d, rows=%d, 首条=%s" % (version, len(rows), rows[0][0]))
    print("提示   : 重新编译 APK 后生效（assets/prompt_emb.bin）")
    if args.mock:
        print("警告   : mock 向量与真实 CLIP 无对应关系，只适合验证链路，不要用于真实检测。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
