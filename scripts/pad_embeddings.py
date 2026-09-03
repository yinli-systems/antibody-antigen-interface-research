import os
import glob
import torch

directory = "/ssd/scxi253/model_deployment/data/interaction_first_asep/embeddings/"
files = glob.glob(directory + "*/*.pt")

fixed = 0
for path in files:
    try:
        data = torch.load(path, map_location="cpu")
        changed = False
        new_chunks = []
        for chunk in data["chunks"]:
            if chunk.shape[1] == 512:
                # Pad to 1280
                pad_size = 1280 - 512
                padded = torch.nn.functional.pad(chunk, (0, pad_size))
                new_chunks.append(padded)
                changed = True
            elif chunk.shape[1] != 1280:
                print(f"Warning: {path} has dimension {chunk.shape[1]}")
                new_chunks.append(chunk)
            else:
                new_chunks.append(chunk)
        if changed:
            data["chunks"] = new_chunks
            torch.save(data, path)
            fixed += 1
    except Exception as e:
        print(f"Failed on {path}: {e}")

print(f"Fixed {fixed} files out of {len(files)} total!")
