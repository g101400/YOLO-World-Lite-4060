#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-World 文本编码器服务 (route A — 开放词汇)
=============================================

YOLO-World 把"检测 80 类"升级为"检测任意文本提示词"。
本服务把用户给出的类别提示词编码成 CLIP 文本嵌入，供 Android 端 native 解码器
（yoloworld.cpp）做 视觉嵌入 · 文本嵌入 的相似度打分。

用法：
    # 1) 无依赖的演示模式（确定性 mock 嵌入，仅用于打通链路/验证 UI）
    python3 yoloworld_text_tower.py --mock

    # 2) 真实 CLIP 文本塔（需 pip install git+https://github.com/openai/CLIP.git torch）
    #    官方 YOLO-World 权重使用的就是 CLIP 文本塔(ViT-B/32, 512 维)，与之对齐效果最佳
    python3 yoloworld_text_tower.py --clip

    # 手机访问：把下面打印出的局域网地址填进 App 的"文本编码器地址"
    # 模拟器默认 http://10.0.2.2:8000

接口：
    POST /embed  body: {"prompts": ["person","car",...]}
    -> 200 {"dim": 512, "embeddings": [[...], ...]}   # 行优先, 已 L2 归一化
    GET  /health -> {"status":"ok","mode":"clip|mock","dim":512}
"""
import argparse
import hashlib
import json
import math
import os
import random
from http.server import BaseHTTPRequestHandler, HTTPServer

EMBED_DIM = 512  # OpenAI CLIP ViT-B/32


def _mock_embedding(text, dim=EMBED_DIM):
    """确定性：同一文本 -> 同一向量；不同文本 -> 不同向量（L2 归一化）。"""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _clip_embeddings(prompts, dim=EMBED_DIM):
    import clip
    import torch
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    with torch.no_grad():
        tok = clip.tokenize(prompts, truncate=True).to(device)
        feats = model.encode_text(tok)              # [N, 512]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        arr = feats.cpu().numpy().astype("float32")
    return arr.tolist(), arr.shape[1]


class Handler(BaseHTTPRequestHandler):
    mode = "mock"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok", "mode": Handler.mode, "dim": EMBED_DIM})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/embed"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
            prompts = [str(p).strip() for p in req.get("prompts", []) if str(p).strip()]
            if not prompts:
                self._send(400, {"error": "empty prompts"})
                return
            if Handler.mode == "clip":
                embs, dim = _clip_embeddings(prompts)
            else:
                dim = EMBED_DIM
                embs = [_mock_embedding(p, dim) for p in prompts]
            self._send(200, {"dim": dim, "embeddings": embs})
        except Exception as e:  # noqa
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # silence default logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true", help="使用真实 CLIP 文本塔(需 torch+clip)")
    ap.add_argument("--mock", action="store_true", help="使用确定性 mock 嵌入(默认)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.clip and not args.mock:
        try:
            import clip  # noqa
            Handler.mode = "clip"
            print("[text-tower] 使用真实 CLIP 文本塔 (ViT-B/32, %d 维)" % EMBED_DIM)
        except Exception as e:
            print("[text-tower] CLIP 不可用(%s)，回退 mock 模式" % e)
            Handler.mode = "mock"
    else:
        Handler.mode = "mock"
        print("[text-tower] 演示模式：确定性 mock 嵌入 (%d 维)" % EMBED_DIM)

    srv = HTTPServer((args.host, args.port), Handler)
    print("[text-tower] 监听 http://%s:%d" % (args.host, args.port))
    print("[text-tower] 手机填: http://<本机局域网IP>:%d  (模拟器填 http://10.0.2.2:%d)" % (args.port, args.port))
    print("[text-tower] 测试: curl -XPOST http://127.0.0.1:%d/embed -d '{\"prompts\":[\"person\",\"car\"]}'" % args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[text-tower] 已停止")


if __name__ == "__main__":
    main()
