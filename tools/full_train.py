"""Retrain a 5-fold tag on all train+val data (--merge_val), same hyperparameters, tag full_<tag>."""
import sys, json, glob, subprocess, os
SEED = os.environ.get("FULL_SEED"); SUF = f"_s{SEED}" if SEED else ""
for tag in sys.argv[1:]:
    cfg = json.load(open(sorted(glob.glob(f"runs/*f0of5_{tag}"))[-1] + "/config.json"))
    args = []
    for k, v in cfg.items():
        if k in ("folds", "fold", "merge_val", "tag", "out", "split_seed") or v is None: continue
        if k == "seed" and SEED: v = SEED
        if isinstance(v, bool):
            if v: args.append(f"--{k}")
        else: args += [f"--{k}", str(v)]
    args += ["--merge_val", "--tag", f"full_{tag}{SUF}"]
    print(" ".join(args), flush=True)
    with open(f"runs/full_{tag}{SUF}.log", "w") as log:
        r = subprocess.run([sys.executable, "train.py", *args], stdout=log, stderr=subprocess.STDOUT)
    print(f"{tag}: exit {r.returncode}", flush=True)
    if r.returncode: break
