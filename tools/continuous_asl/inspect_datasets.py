"""Inspect acquired metadata without inventing glosses or accessing protected payloads."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_how2sign(root: Path) -> dict:
    sets = {}
    source_ids = {}
    for split in ("train", "val"):
        path = root / "data/raw/how2sign" / ("how2sign_realigned_" + split + ".csv")
        if not path.exists():
            sets[split] = dict(status="NOT_ACQUIRED")
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"VIDEO_ID", "SENTENCE_ID", "SENTENCE_NAME", "START_REALIGNED", "END_REALIGNED", "SENTENCE"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("HOW2SIGN_TRANSLATION_SCHEMA")
            rows = list(reader)
        invalid = empty = 0
        ids = []
        seconds = []
        for row in rows:
            start, end = float(row["START_REALIGNED"]), float(row["END_REALIGNED"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                invalid += 1
            else:
                seconds.append(end-start)
            empty += not bool(row["SENTENCE"].strip())
            ids.append(row["SENTENCE_NAME"])
        source_ids[split] = {r["VIDEO_ID"] for r in rows}
        sets[split] = dict(status="METADATA_VALIDATED", rows=len(rows),
                           source_videos=len(source_ids[split]), duplicate_sentence_names=len(ids)-len(set(ids)),
                           invalid_timestamps=invalid, empty_translations=empty,
                           annotated_seconds=sum(seconds), sha256=sha(path),
                           annotation_type="ENGLISH_TRANSLATION_NOT_GLOSS",
                           video_pairing_verified=False, gloss_ctc_eligible=False)
    overlap = source_ids.get("train", set()) & source_ids.get("val", set())
    return dict(splits=sets, train_val_source_video_overlap=len(overlap),
                payload_test_accessed=False, authentic_gloss_download_found=False)


def inspect_ncslgr(root: Path) -> tuple[dict, list[dict]]:
    path = next((root/"data/extracted/ncslgr_video_index").rglob("files_by_xml_file.csv"))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identities = defaultdict(set)
    for row in rows:
        identities[row["Compressed MOV file"]].add((row["XML file"], row["Occurs in utterance id"], row["main gloss"]))
    rejected = Counter()
    eligible = []
    seen = set()
    # Deliberately restrictive lexical subset. Preserve original gloss tokens verbatim.
    # Quotes, classifier descriptions, fingerspelled strings and uncertain labels are excluded.
    lexical = re.compile(r"(?:\(2h\))?[A-Z][A-Z0-9_-]*(?:\+{1,2})?$")
    for row in rows:
        url = row["Compressed MOV file"]
        identity = (row["XML file"], row["Occurs in utterance id"])
        if row["Perspective/Camera id"] != "0":
            rejected["other_camera"] += 1
            continue
        if len(identities[url]) != 1:
            rejected["video_reused_across_utterances"] += 1
            continue
        parts = urlsplit(url)
        if parts.hostname != "csr.bu.edu" or not parts.path.endswith(".mov"):
            rejected["not_small_per_utterance_source"] += 1
            continue
        tokens = row["main gloss"].split()
        if not 2 <= len(tokens) <= 12 or any(not lexical.fullmatch(x) for x in tokens):
            rejected["complex_or_fingerspelled_gloss"] += 1
            continue
        if identity in seen:
            continue
        seen.add(identity)
        key = hashlib.sha256((identity[0]+"|"+identity[1]).encode()).hexdigest()[:16]
        eligible.append(dict(sample_id="ncslgr_"+key, dataset="NCSLGR",
                             collection=identity[0], utterance_id=identity[1],
                             source_group=row["Video sequence id"], signer_id=None,
                             signer_status="NOT_AVAILABLE_IN_PUBLIC_INDEX",
                             source_url_original=url, source_url=urlunsplit(("https", parts.netloc, parts.path, parts.query, "")),
                             gloss_text=row["main gloss"], glosses=tokens,
                             annotation_type="AUTHENTIC_GLOSS_SEQUENCE_FROM_OFFICIAL_VIDEO_INDEX",
                             individual_sign_boundaries_available=False,
                             license_reference="https://www.bu.edu/asllrp/dai-terms.html"))
    report = dict(rows=len(rows), collections=len({r["XML file"] for r in rows}),
                  unique_video_urls=len(identities), eligible_lexical_utterances=len(eligible),
                  rejection_rows=dict(rejected), index_sha256=sha(path),
                  signer_independent_split_possible_from_index_alone=False,
                  xml_download_requires_free_dai_account=True,
                  old_signstream_archive_safe_extraction="BLOCKED_WINDOWS_TRAILING_SPACE_NAMES",
                  restrictions="PRIVATE_NONCOMMERCIAL_RESEARCH; NO REDISTRIBUTION OR MODEL_DISTRIBUTION_RIGHTS_ASSERTED")
    return report, sorted(eligible, key=lambda x: x["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    report, eligible = inspect_ncslgr(root)
    result = dict(schema="apphear.continuous_asl.metadata_quality.v1",
                  how2sign=inspect_how2sign(root), ncslgr=report)
    (root/"metadata_quality.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    (root/"ncslgr_eligible_index.json").write_text(json.dumps(eligible, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))
if __name__ == "__main__":
    main()

