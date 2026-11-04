"""Fetch a pinned public feature archive and prepare train-only text features."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import torch

REVISION = "bf56d4f6b7a191b393e50894b81ee4089a4131f4"
URL = ("https://huggingface.co/datasets/RichyHu999/multimodal-teacher-data/resolve/"
       + REVISION + "/msrvtt.zip")
ARCHIVE_SHA256 = "bb8cc603f9cfd0543dfa03bc88a4dae887f97ffcd21a59fc97aecc2296311828"
CATEGORIES = ["music", "people", "gaming", "sports/actions", "news/events/politics",
              "education", "tv shows", "movie/comedy", "animation", "vehicles/autos",
              "howto", "travel", "science/technology", "animals/pets", "kids/family",
              "documentary", "food/drink", "cooking", "beauty/fashion", "advertisement"]


def fit_source_text(texts, source_mask, max_features=2000):
    """Vocabulary and nonzero text rows are restricted to the source train set."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    source_rows = vectorizer.fit_transform(np.asarray(texts)[source_mask])
    result = np.zeros((len(texts), len(vectorizer.vocabulary_)), dtype=np.float32)
    result[source_mask] = source_rows.toarray().astype(np.float32)
    return result, vectorizer.get_feature_names_out().tolist()


def assign_splits(video_ids, train_ids, val_ids, test_ids):
    groups = [set(train_ids), set(val_ids), set(test_ids)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Published split lists overlap")
    if set.union(*groups) != set(video_ids):
        raise ValueError("Published split lists do not cover the feature IDs exactly")
    lookup = {v: split for group, split in zip(groups, ["train", "validate", "test"]) for v in group}
    return np.asarray([lookup[v] for v in video_ids])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    downloads = args.data_root / "downloads"
    prepared = args.data_root / "prepared"
    downloads.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    archive = downloads / "msrvtt.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print("Downloading", URL, flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Archive checksum mismatch: {digest}")

    with ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        annotation = json.loads(z.read("msrvtt/Annotations/MSRVTT_data.json"))
        rows = sorted(annotation["videos"], key=lambda x: int(x["video_id"][5:]))
        captions = defaultdict(list)
        for item in annotation["sentences"]:
            captions[item["video_id"]].append(item["caption"])
        ids = [r["video_id"] for r in rows]
        def split_ids(filename):
            contents = z.read("msrvtt/Annotations/" + filename).decode()
            return {r["video_id"] for r in csv.DictReader(io.StringIO(contents))}
        splits = assign_splits(ids, split_ids("MSRVTT_train.7k.csv"),
                               split_ids("MSRVTT_full_split_val.csv"),
                               split_ids("MSRVTT_full_split_test.csv"))
        if dict(Counter(splits.tolist())) != {"train": 6513, "validate": 497, "test": 2990}:
            raise ValueError("Unexpected Full split sizes")
        features = []
        feature_hashes = {}
        for i, video_id in enumerate(ids):
            name = f"msrvtt/FeatureData/ImageBind/VideoFeature/{video_id}.pt"
            b = z.read(name)
            x = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
            if not torch.is_tensor(x) or tuple(x.shape) != (1024,):
                raise ValueError(f"Unexpected tensor in {name}")
            a = x.detach().numpy().astype(np.float32)
            if not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12:
                raise ValueError(f"Invalid visual feature: {video_id}")
            features.append(a)
            feature_hashes[video_id] = hashlib.sha256(b).hexdigest()
            if (i + 1) % 2000 == 0:
                print(f"Verified {i + 1}/{len(ids)} visual tensors", flush=True)
    visual = np.stack(features)
    labels = np.asarray([r["category"] for r in rows], dtype=np.int64)
    texts = [" ".join(captions[video_id]) for video_id in ids]
    if len(set(ids)) != 10000 or any(len(captions[v]) != 20 for v in ids):
        raise ValueError("Expected 10,000 unique videos with 20 descriptions each")
    text_features, vocabulary = fit_source_text(texts, splits == "train")

    duplicates = defaultdict(list)
    for video_id, split, x in zip(ids, splits, visual):
        duplicates[hashlib.sha256(x.tobytes()).hexdigest()].append((video_id, str(split)))
    cross_split_duplicates = [v for v in duplicates.values() if len({x[1] for x in v}) > 1]
    # Keep the published split intact, but make exact feature duplicates visible.
    manifest = {
        "dataset": "MSR-VTT", "feature_archive_url": URL,
        "feature_archive_sha256": digest, "feature_mirror_revision": REVISION,
        "visual_feature": "public ImageBind video-feature tensors, 1024 dimensions",
        "provenance_limit": "Third-party mirror preserves bytes; original extraction settings are not independently established.",
        "text_feature": "CountVectorizer, English stop words, top 2000 training words; raw counts over all 20 source captions",
        "text_vocabulary_fit_split": "train", "target_text_features": "zero, never constructed from target captions",
        "splits": dict(Counter(splits.tolist())),
        "split_source": "Explicit Full split CSVs; the retrieval JSON marks every video as train and is NOT used for splits.",
        "category_names": CATEGORIES,
        "class_counts": {str(c): {s: int(np.sum((labels == c) & (splits == s)))
                                  for s in ["train", "validate", "test"]}
                         for c in range(20)},
        "cross_split_exact_feature_duplicates": cross_split_duplicates,
        "videos": len(ids), "captions": sum(map(len, captions.values())),
        "scope": "New controlled MIL-PI classification protocol; not the original IJCV 2016 feature pipeline or published table numbers."
    }
    np.savez_compressed(prepared / "msrvtt.npz", visual=visual, text=text_features,
                        labels=labels, splits=splits, video_ids=np.asarray(ids))
    (prepared / "msrvtt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (prepared / "msrvtt_vocabulary.json").write_text(json.dumps(vocabulary, indent=2) + "\n")
    (prepared / "visual_feature_sha256.json").write_text(json.dumps(feature_hashes, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
