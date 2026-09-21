#!/usr/bin/env python3
"""
X1331 LIVE-08E.0 — Prospective sequence builder / validator.

Reads LIVE-08D append-only causal examples and creates one sequence per
independent network outcome. It DOES NOT train a model and DOES NOT feed
outcome information into BEFORE features.

Outputs:
  data/live08e-sequences.jsonl
  data/live08e-manifest.json
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

SCHEMA = "x1331-live08e-sequence-v1"
STATE_NAMES = ["000", "001", "010", "011", "100", "101", "110", "111"]


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception as e:
                raise SystemExit(f"JSON error line {line_no}: {e}")
            r["_source_line"] = line_no
            rows.append(r)
    return rows


def validate_row(r):
    b = r["before"]
    o = r["outcome"]
    c = b["cognition"]

    assert r["relation"] == "DIRECT_SUCCESSOR"
    assert b["prevhash_canonical"] == o["previousblockhash"]
    assert b["frozen_at"] <= o["observed_at"]

    for key in ("probabilities", "evidence", "observations"):
        assert len(c[key]) == 8, (b["freeze_id"], key)

    assert all(math.isfinite(float(x)) for x in c["probabilities"])
    assert all(math.isfinite(float(x)) for x in c["evidence"])
    assert abs(sum(float(x) for x in c["probabilities"]) - 1.0) < 1e-9

    leader = c.get("leader")
    if leader is not None:
        assert leader in STATE_NAMES

    return b, o, c


def before_vector(b, c):
    # ONLY fields known at freeze time.
    probs = [float(x) for x in c["probabilities"]]
    evidence = [float(x) for x in c["evidence"]]
    observations = [float(x) for x in c["observations"]]

    leader = c.get("leader")
    leader_onehot = [1.0 if leader == s else 0.0 for s in STATE_NAMES]

    # Pool difficulty is pre-outcome context. Log transform keeps scale sane.
    diff = max(float(b.get("pool_difficulty", 0.0)), 0.0)

    return (
        probs
        + evidence
        + observations
        + leader_onehot
        + [
            float(c["recognition"]),
            float(c["stability"]),
            float(c["confidence_internal"]),
            float(c["leader_streak"]),
            float(c["cognitive_cycles"]),
            float(c["incoming"]),
            math.log2(diff + 1.0),
        ]
    )


def outcome_label(o):
    # Label/metadata stays strictly outside BEFORE feature vectors.
    nonce = int(o["nonce"])
    return {
        "height": int(o["height"]),
        "hash": o["hash"],
        "previousblockhash": o["previousblockhash"],
        "version": int(o["version"]),
        "timestamp": int(o["timestamp"]),
        "mediantime": int(o["mediantime"]),
        "nonce": nonce,
        "nonce_state": format((nonce >> 29) & 0x7, "03b"),
        "bits": int(o["bits"]),
        "difficulty": float(o["difficulty"]),
        "tx_count": int(o["tx_count"]),
        "size": int(o["size"]),
        "weight": int(o["weight"]),
        "observed_at": int(o["observed_at"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/live08d-prospective.jsonl")
    ap.add_argument("--output", default="data/live08e-sequences.jsonl")
    ap.add_argument("--manifest", default="data/live08e-manifest.json")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    manifest_path = Path(args.manifest)

    rows = load_rows(inp)
    if not rows:
        raise SystemExit("No LIVE-08D prospective examples yet.")

    groups = defaultdict(list)
    for r in rows:
        b, o, c = validate_row(r)
        groups[(int(o["height"]), o["hash"])].append((r, b, o, c))

    sequences = []
    seen_freezes = set()

    for (height, block_hash), items in sorted(groups.items()):
        items.sort(key=lambda x: (int(x[1]["frozen_at"]), int(x[1]["freeze_id"])))
        first_o = items[0][2]

        # All rows in one sequence must resolve to exactly the same outcome.
        for _, b, o, _ in items:
            assert int(o["height"]) == height
            assert o["hash"] == block_hash
            assert o["previousblockhash"] == first_o["previousblockhash"]
            fid = int(b["freeze_id"])
            assert fid not in seen_freezes, f"duplicate freeze_id {fid}"
            seen_freezes.add(fid)

        steps = []
        for _, b, _, c in items:
            vec = before_vector(b, c)
            steps.append(
                {
                    "freeze_id": int(b["freeze_id"]),
                    "frozen_at": int(b["frozen_at"]),
                    "incoming": int(b["incoming"]),
                    "job_id": b["job_id"],
                    "prevhash_canonical": b["prevhash_canonical"],
                    "pool_difficulty": float(b["pool_difficulty"]),
                    "leader": c.get("leader"),
                    "recognition": float(c["recognition"]),
                    "stability": float(c["stability"]),
                    "confidence_internal": float(c["confidence_internal"]),
                    "leader_streak": int(c["leader_streak"]),
                    "cognitive_cycles": int(c["cognitive_cycles"]),
                    "probabilities": [float(x) for x in c["probabilities"]],
                    "evidence": [float(x) for x in c["evidence"]],
                    "observations": [int(x) for x in c["observations"]],
                    "features": vec,
                }
            )

        sequences.append(
            {
                "schema": SCHEMA,
                "sequence_id": f"btc-{height}-{block_hash[:16]}",
                "parent_hash": first_o["previousblockhash"],
                "length": len(steps),
                "feature_dim": len(steps[0]["features"]),
                "steps": steps,
                "label": outcome_label(first_o),
            }
        )

    # Rebuild output deterministically from the current immutable closed examples.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for seq in sequences:
            json.dump(seq, f, separators=(",", ":"), sort_keys=True)
            f.write("\n")

    lengths = [s["length"] for s in sequences]
    feature_dims = sorted({s["feature_dim"] for s in sequences})
    assert len(feature_dims) == 1

    manifest = {
        "schema": SCHEMA,
        "source": str(inp),
        "source_sha256_at_build": sha256_file(inp),
        "output": str(out),
        "output_sha256": sha256_file(out),
        "examples": len(rows),
        "independent_outcomes": len(sequences),
        "feature_dim": feature_dims[0],
        "sequence_lengths": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "state_names": STATE_NAMES,
        "feature_contract": {
            "before_only": True,
            "probabilities": 8,
            "evidence": 8,
            "observations": 8,
            "leader_onehot": 8,
            "scalars": [
                "recognition",
                "stability",
                "confidence_internal",
                "leader_streak",
                "cognitive_cycles",
                "incoming",
                "log2_pool_difficulty_plus_1",
            ],
            "outcome_fields_in_features": [],
        },
        "split_policy": (
            "Chronological by independent outcome only. "
            "Never split steps from the same block across train/validation/holdout."
        ),
        "training_ready": len(sequences) >= 3,
        "scientific_note": (
            "training_ready only means enough independent groups to exercise a split; "
            "it does not imply enough data for a meaningful predictive claim."
        ),
    }
    atomic_json(manifest_path, manifest)

    print("X1331 LIVE-08E.0")
    print("PROSPECTIVE SEQUENCE BUILDER")
    print(f"source examples      = {len(rows)}")
    print(f"independent outcomes = {len(sequences)}")
    print(f"feature dimension    = {feature_dims[0]}")
    print(f"sequence lengths     = {lengths}")
    print(f"output               = {out}")
    print(f"output sha256        = {manifest['output_sha256']}")
    print("BEFORE-ONLY FEATURES = PASS")
    print("BLOCK GROUPING       = PASS")
    print("NO CROSS-BLOCK SPLIT = ENFORCED")
    if len(sequences) < 3:
        print("MODEL TRAINING       = WAIT (need more independent outcomes)")
    else:
        print("MODEL TRAINING       = structurally available; statistical adequacy still unproven")


if __name__ == "__main__":
    main()
