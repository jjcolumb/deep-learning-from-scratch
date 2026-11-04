"""
Build the frog / not-frog photo set from CIFAR-10.

CIFAR-10 is 60,000 tiny photos (32 x 32 pixels) in 10 classes. We keep every
frog and the same number of other photos (the first ones in file order, which
CIFAR already shuffled), so a coin flip scores 50%.

    python make_frogs.py                         # downloads CIFAR-10 (170 MB)
    python make_frogs.py cifar-10-binary.tar.gz  # or use a copy you have

Writes frogs_train.bin (5,000 frogs + 5,000 others) and frogs_test.bin
(1,000 + 1,000), in CIFAR's own binary layout: one byte for the label
(1 = frog, 0 = not), then 3,072 bytes of pixels (1,024 red, 1,024 green,
1,024 blue, row by row). No library needed to read them.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
HERE = Path(__file__).parent
FROG = 6                                   # CIFAR-10 class number for frogs
RECORD = 1 + 3072

archive = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "cifar-10-binary.tar.gz"
if not archive.exists():
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, archive)


def pick(blobs, frogs_wanted):
    """Every frog, plus the first equal number of non-frogs, in file order."""
    out, frogs, others = bytearray(), 0, 0
    for blob in blobs:
        for at in range(0, len(blob), RECORD):
            label, pixels = blob[at], blob[at + 1:at + RECORD]
            if label == FROG:
                frogs += 1
            elif others < frogs_wanted:
                others += 1
            else:
                continue
            out.append(1 if label == FROG else 0)
            out += pixels
    assert frogs == others == frogs_wanted, (frogs, others)
    return out


with tarfile.open(archive) as tar:
    read = lambda name: tar.extractfile(f"cifar-10-batches-bin/{name}").read()
    train = pick([read(f"data_batch_{i}.bin") for i in range(1, 6)], 5000)
    test = pick([read("test_batch.bin")], 1000)

(HERE / "frogs_train.bin").write_bytes(train)
(HERE / "frogs_test.bin").write_bytes(test)
print(f"frogs_train.bin: {len(train) // RECORD} photos, frogs_test.bin: {len(test) // RECORD} photos")
