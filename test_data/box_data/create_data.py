import os

import mrcfile
import numpy as np

OUTPUT_DIR = os.path.dirname(__file__)
NDIM = 50
np.random.seed(42)

for mic_id in range(10):
    data = np.zeros((NDIM, NDIM), dtype=np.float32)
    coords = []
    for box_id in range(3):
        x_coord = int(np.random.random() * NDIM)
        y_coord = int(np.random.random() * NDIM)
        coords.append((y_coord, x_coord, 0, 0))
        data[x_coord, y_coord] = 1

    with mrcfile.new(
        os.path.join(OUTPUT_DIR, f"test_{mic_id}.mrc"), overwrite=True
    ) as new:
        new.set_data(data)

    for suffix in ("", "_centered", "_new"):
        if np.random.random() > 0.8:
            continue

        with open(f"test_{mic_id}{suffix}.box", "w") as write:
            write.write(
                "\n".join([" ".join(map(str, entry)) for entry in coords])
            )
