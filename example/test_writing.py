from pathlib import Path

p = Path("results/foo.txt")

if p.exists():
    # the file already exists and I just write new results
    with p.open("w") as f:
        f.write("new results")
else:
    # the file does not exist, but maybe the parent does ...
    parent = p.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=False)
    with p.open("w") as f:
        f.write("initial results")
