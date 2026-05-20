"""Download helpers for public experiment sources."""

from __future__ import annotations

import json
from pathlib import Path

import requests
from tqdm import tqdm


def stream_download(url: str, path: str | Path, *, chunk_size: int = 1024 * 1024) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = 0
    if tmp.exists():
        existing = tmp.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, timeout=60, headers=headers) as resp:
        if resp.status_code == 416:
            tmp.rename(path)
            return path
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", "0") or 0)
        if resp.status_code == 206:
            total += existing
        with open(tmp, mode) as f, tqdm(
            total=total or None,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=path.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(path)
    return path


def download_cellxgene_assets(metadata_path: str | Path, titles: list[str], out_dir: str | Path):
    metadata = json.loads(Path(metadata_path).read_text())
    out_dir = Path(out_dir)
    downloaded = []
    for dataset in metadata.get("datasets", []):
        title = dataset.get("title", "")
        if title not in titles:
            continue
        assets = [a for a in dataset.get("assets", []) if a.get("filetype") == "H5AD"]
        if not assets:
            continue
        url = assets[0]["url"]
        name = title.lower().replace(" ", "_").replace("-", "_") + ".h5ad"
        downloaded.append(stream_download(url, out_dir / name))
    return downloaded
