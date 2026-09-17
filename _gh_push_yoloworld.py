#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 YOLO-World-Lite 工程推送到 GitHub (g101400/YOLO-World-Lite-4060)。
通过 GitHub REST / Git Database API 提交（不依赖 git 传输协议，适配本沙箱网络）。
用法: GH_TOKEN=xxxx python3 _gh_push_yoloworld.py
"""
import base64
import fnmatch
import json
import os
import sys
import urllib.request

API = "https://api.github.com"
TOKEN = os.environ.get("GH_TOKEN")
if not TOKEN:
    print("ERROR: 请设置 GH_TOKEN 环境变量"); sys.exit(1)

OWNER = "g101400"
REPO = "YOLO-World-Lite-4060"
ROOT = os.path.dirname(os.path.abspath(__file__))
BRANCH = "main"

# ---- gitignore (简化匹配) ----
IGNORE = [
    "*.iml", ".gradle", "/local.properties", "/.idea", ".DS_Store",
    "/build", "/app/build", "/captures", ".externalNativeBuild", "build.log",
    "verify.log", "push.log", ".cxx", "*.apk",
    "keystore.properties", "release-key.jks",
    "app/src/main/assets/*.param", "app/src/main/assets/*.bin",
]


def is_ignored(rel):
    parts = rel.replace("\\", "/").split("/")
    for pat in IGNORE:
        p = pat.lstrip("/")
        if p.endswith("/"):
            p = p[:-1]
        if "/" in p:
            if fnmatch.fnmatch(rel.replace("\\", "/"), p):
                return True
        else:
            for part in parts:
                if fnmatch.fnmatch(part, p):
                    return True
    return False


def api(method, path, data=None):
    url = API + path
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", "token " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "yoloworld-push")
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        return e.code, body


def main():
    # 1) 建仓库
    st, _ = api("POST", "/user/repos", {
        "name": REPO, "description": "YOLO-World 开放词汇检测 (路线A) Android ncnn demo",
        "private": False, "auto_init": False,
    })
    if st not in (201, 422):
        print("创建仓库失败:", st); sys.exit(1)
    print("仓库就绪:", OWNER + "/" + REPO)

    # 1.5) 空仓库没有默认分支，Git Database API 建 blob 会 409；先用 contents API 种一个初始提交
    st_init, js_init = api("PUT", "/repos/%s/%s/contents/README.md" % (OWNER, REPO), {
        "message": "init repository",
        "content": base64.b64encode(b"# YOLO-World-Lite-4060\n").decode("ascii"),
    })
    print("初始化默认分支:", st_init, (js_init if st_init >= 400 else "ok"))

    # 2) 收集文件
    files = []
    for dp, dn, fns in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", "build", ".gradle", ".idea", "captures", ".externalNativeBuild", ".cxx")]
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, ROOT)
            if is_ignored(rel):
                continue
            if os.path.getsize(full) > 25 * 1024 * 1024:
                continue  # 跳过过大文件
            files.append((rel, full))
    # 保证根目录文件在前
    files.sort()
    print("待提交文件数:", len(files))

    # 3) 创建 blobs
    blobs = {}
    for rel, full in files:
        with open(full, "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        st, js = api("POST", "/repos/%s/%s/git/blobs" % (OWNER, REPO),
                     {"content": content, "encoding": "base64"})
        if st != 201:
            print("blob 失败", rel, st, js); sys.exit(1)
        blobs[rel] = js["sha"]

    # 4) 创建 tree
    tree = [{"path": rel.replace("\\", "/"), "mode": "100644",
             "type": "blob", "sha": blobs[rel]} for rel, _ in files]
    st, js = api("POST", "/repos/%s/%s/git/trees" % (OWNER, REPO), {"tree": tree})
    if st != 201:
        print("tree 失败", st, js); sys.exit(1)
    tree_sha = js["sha"]

    # 5) 取当前 HEAD 作为父提交（保证 fast-forward，普通 PATCH 即可推进）
    parent_sha = None
    st, js = api("GET", "/repos/%s/%s/git/refs/heads/%s" % (OWNER, REPO, BRANCH))
    if st == 200 and isinstance(js, dict):
        parent_sha = js.get("object", {}).get("sha")
    parents = [parent_sha] if parent_sha else []
    print("父提交:", parent_sha or "(无, 首次提交)")

    st, js = api("POST", "/repos/%s/%s/git/commits" % (OWNER, REPO), {
        "message": "YOLO-World-Lite 开放词汇检测 demo (路线A)\n\n- ncnn Android demo: 文本提示词驱动开放词汇检测\n- YOLOv8 同族解码器 + 文本嵌入相似度分类\n- 零依赖 ywmimage 渲染(去除 OpenCV 链接，解决 NDK ABI 不匹配)\n- Gradle 自动签名 release APK\n- CLIP 文本塔服务(真实/mock) + 模型转换脚本 + 算法自测",
        "tree": tree_sha, "parents": parents,
    })
    if st != 201:
        print("commit 失败", st, js); sys.exit(1)
    commit_sha = js["sha"]

    # 6) 创建或推进 ref
    if parent_sha:
        st, js = api("PATCH", "/repos/%s/%s/git/refs/heads/%s" % (OWNER, REPO, BRANCH),
                     {"sha": commit_sha, "force": True})
        print("update ref:", st, ("" if st == 200 else str(js)[:200]))
    else:
        st, js = api("POST", "/repos/%s/%s/git/refs" % (OWNER, REPO),
                     {"ref": "refs/heads/" + BRANCH, "sha": commit_sha})
        print("create ref:", st, ("" if st == 201 else str(js)[:200]))
    print("PUSH 完成 ✅  https://github.com/%s/%s" % (OWNER, REPO))


if __name__ == "__main__":
    main()
