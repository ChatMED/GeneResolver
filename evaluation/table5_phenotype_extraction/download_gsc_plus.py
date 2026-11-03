"""Download the original GSC+ corpus (228 annotated abstracts)."""
from __future__ import annotations

import subprocess
import tempfile
import urllib.request
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "gsc_plus"
SOURCE = "https://raw.githubusercontent.com/lasigeBioTM/IHP/master/GSC%2B.rar"
SOURCE_SHA256 = "59e57552543528cf9b980cd48abab9b43fae22adc191e7a3db539770f2a7b2e79"


def main() -> None:
    annotations = DEST / "Annotations"
    texts = DEST / "Text"
    if annotations.is_dir() and texts.is_dir():
        if len(list(annotations.iterdir())) == len(list(texts.iterdir())) == 228:
            print(f"GSC+ ready: {DEST} (228 paired documents)")
            return

    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DEST) as temporary:
        archive = Path(temporary) / "GSC+.rar"
        print(f"Downloading {SOURCE}")
        urllib.request.urlretrieve(SOURCE, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError("GSC+ archive checksum differs from the verified source")
        names = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
        if any(name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name
               for name in names):
            raise RuntimeError("GSC+ archive contains an unsafe path")
        subprocess.run(["tar", "-xf", str(archive), "-C", str(DEST)], check=True)

    if len(list(annotations.iterdir())) != 228 or len(list(texts.iterdir())) != 228:
        raise RuntimeError("GSC+ download did not contain 228 paired documents")
    print(f"GSC+ ready: {DEST} (228 paired documents)")


if __name__ == "__main__":
    main()
