"""Bounded archive inspection/extraction; no source deletion or existing-file overwrite."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


def member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized or normalized.startswith("/") or ":" in normalized
            or ".." in path.parts or "\x00" in normalized):
        raise ValueError("UNSAFE_ARCHIVE_PATH")
    for component in path.parts:
        if component.endswith((" ", ".")) or component.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *("COM%d" % n for n in range(1, 10)),
            *("LPT%d" % n for n in range(1, 10))
        }:
            raise ValueError("WINDOWS_UNSAFE_ARCHIVE_PATH")
    return path


def assert_disjoint(sources: list[Path], destination: Path) -> None:
    out = destination.resolve()
    for source in sources:
        src = source.resolve()
        if src == out or src in out.parents or out in src.parents:
            raise ValueError("SOURCE_OUTPUT_OVERLAP")


def inspect_archive(path: Path, max_bytes: int, max_entries: int = 10000) -> list[dict]:
    rows: list[dict] = []
    names: set[str] = set()
    def add(name: str, size: int, directory: bool) -> None:
        safe = str(member_path(name))
        key = safe.casefold()
        if key in names:
            raise ValueError("DUPLICATE_ARCHIVE_PATH")
        names.add(key)
        rows.append(dict(path=safe, bytes=size, directory=directory))
        if len(rows) > max_entries or sum(r["bytes"] for r in rows) > max_bytes:
            raise ValueError("ARCHIVE_BUDGET_EXCEEDED")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("ARCHIVE_LINK_OR_SPECIAL_FILE")
                if info.flag_bits & 1:
                    raise ValueError("ENCRYPTED_ARCHIVE_NOT_SUPPORTED")
                if info.file_size > max(1, info.compress_size) * 1000:
                    raise ValueError("ARCHIVE_EXPANSION_RATIO")
                add(info.filename, info.file_size, info.is_dir())
            bad = archive.testzip()
            if bad:
                raise ValueError("ZIP_CRC_FAILURE")
    else:
        with tarfile.open(path, "r:*") as archive:
            for info in archive:
                if not (info.isfile() or info.isdir()):
                    raise ValueError("ARCHIVE_LINK_OR_SPECIAL_FILE")
                add(info.name, info.size if info.isfile() else 0, info.isdir())
    return rows


def extract(path: Path, destination: Path, allowed_root: Path, max_bytes: int) -> dict:
    rows = inspect_archive(path, max_bytes)
    root, target = allowed_root.resolve(), destination.resolve()
    if root not in target.parents:
        raise ValueError("EXTRACTION_ESCAPES_ALLOWED_ROOT")
    assert_disjoint([path], target)
    if target.exists():
        raise FileExistsError("EXTRACTION_TARGET_MUST_BE_NEW")
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink():
            raise ValueError("EXTRACTION_REPARSE_POINT")
        if ancestor == root:
            break
    target.mkdir(parents=True)
    def write(name: str, stream, size: int) -> None:
        rel = member_path(name)
        dest = target.joinpath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with dest.open("xb") as output:
            while True:
                block = stream.read(min(1024 * 1024, size - count + 1))
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError("ARCHIVE_SIZE_MISMATCH")
                output.write(block)
        if count != size:
            raise ValueError("ARCHIVE_SIZE_MISMATCH")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    target.joinpath(*member_path(info.filename).parts).mkdir(parents=True, exist_ok=True)
                else:
                    with archive.open(info) as stream:
                        write(info.filename, stream, info.file_size)
    else:
        with tarfile.open(path, "r:*") as archive:
            for info in archive:
                if info.isdir():
                    target.joinpath(*member_path(info.name).parts).mkdir(parents=True, exist_ok=True)
                else:
                    with archive.extractfile(info) as stream:
                        write(info.name, stream, info.size)
    return dict(status="EXTRACTED", files=sum(not x["directory"] for x in rows),
                expanded_bytes=sum(x["bytes"] for x in rows), source_retained=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--max-expanded-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--allowed-root", type=Path)
    args = parser.parse_args()
    if args.extract_to:
        if not args.allowed_root:
            parser.error("--allowed-root is required")
        result = extract(args.archive, args.extract_to, args.allowed_root, args.max_expanded_bytes)
    else:
        rows = inspect_archive(args.archive, args.max_expanded_bytes)
        result = dict(status="ARCHIVE_INSPECTED", entries=len(rows),
                      expanded_bytes=sum(x["bytes"] for x in rows), members=rows)
    print(json.dumps(result))
if __name__ == "__main__":
    main()

