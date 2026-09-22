#!/usr/bin/env python3

import hashlib
import json
from collections import defaultdict
from pathlib import Path

OLD = Path("data/live08d-prospective.jsonl")
NEW = Path("data/live08d1-prospective.jsonl")

OUTPUT = Path("data/live08e2-sequences.jsonl")
MANIFEST = Path("data/live08e2-manifest.json")

STATES = [
    "000", "001", "010", "011",
    "100", "101", "110", "111",
]

FEATURE_DIM = 39


def nonce_state(nonce: int) -> str:
    # Same X1331 state definition used by the previous sequence builder:
    # nonce bits 7..5.
    return f"{((int(nonce) >> 5) & 0b111):03b}"


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

            rows.append(row)

    return rows


def outcome_hash(outcome):
    # 08D used "hash"; 08D.1/Blockstream uses "id".
    value = outcome.get("hash")

    if value is None:
        value = outcome.get("id")

    if not value:
        raise RuntimeError("outcome missing hash/id")

    return str(value)


def build_features(before):
    cognition = before["cognition"]

    probabilities = cognition["probabilities"]
    evidence = cognition["evidence"]
    observations = cognition["observations"]

    if len(probabilities) != 8:
        raise RuntimeError("probabilities != 8")

    if len(evidence) != 8:
        raise RuntimeError("evidence != 8")

    if len(observations) != 8:
        raise RuntimeError("observations != 8")

    leader = cognition.get("leader")

    leader_one_hot = [
        1.0 if leader == state else 0.0
        for state in STATES
    ]

    recognition = float(cognition["recognition"])
    stability = float(cognition["stability"])
    confidence_internal = float(
        cognition["confidence_internal"]
    )
    leader_streak = float(cognition["leader_streak"])
    cognitive_cycles = float(cognition["cognitive_cycles"])
    incoming = float(
        cognition.get(
            "incoming",
            before["incoming"],
        )
    )

    # Preserve the exact 39-D semantic layout from LIVE-08E.0.
    #
    #  0..7   probabilities
    #  8..15  evidence
    # 16..23  observations
    # 24..31  leader one-hot
    # 32      recognition
    # 33      stability
    # 34      confidence_internal
    # 35      leader_streak
    # 36      cognitive_cycles
    # 37      incoming
    # 38      log1p(pool difficulty)
    features = (
        [float(v) for v in probabilities]
        + [float(v) for v in evidence]
        + [float(v) for v in observations]
        + leader_one_hot
        + [
            recognition,
            stability,
            confidence_internal,
            leader_streak,
            cognitive_cycles,
            incoming,
            __import__("math").log1p(
                float(before["pool_difficulty"])
            ),
        ]
    )

    if len(features) != FEATURE_DIM:
        raise RuntimeError(
            f"feature dimension {len(features)} != {FEATURE_DIM}"
        )

    return features


def canonicalize(source_name, row):
    relation = row.get("relation")

    if relation != "DIRECT_SUCCESSOR":
        raise RuntimeError(
            f"{source_name}: unexpected relation={relation!r}"
        )

    before = row["before"]
    outcome = row["outcome"]

    parent = str(before["prevhash_canonical"])
    outcome_parent = str(outcome["previousblockhash"])

    if parent != outcome_parent:
        raise RuntimeError(
            f"{source_name}: causal mismatch "
            f"before.parent={parent} "
            f"outcome.parent={outcome_parent}"
        )

    height = int(outcome["height"])
    nonce = int(outcome["nonce"])

    return {
        "source": source_name,
        "source_schema": row.get("schema"),
        "before": before,
        "outcome": outcome,
        "height": height,
        "parent_hash": parent,
        "block_hash": outcome_hash(outcome),
        "nonce_state": nonce_state(nonce),
    }


def main():
    print("X1331 LIVE-08E.2")
    print("UNIFIED CAUSAL SEQUENCE BUILDER")
    print(
        "08D + 08D.1 | BEFORE-ONLY FEATURES | "
        "ONE BLOCK = ONE OUTCOME"
    )
    print()

    raw = []

    for source_name, path in [
        ("LIVE-08D", OLD),
        ("LIVE-08D.1", NEW),
    ]:
        rows = load_jsonl(path)

        print(f"{source_name:10s} rows = {len(rows)}")

        for row in rows:
            raw.append(canonicalize(source_name, row))

    if not raw:
        raise SystemExit("no prospective rows found")

    # Group all BEFORE snapshots by the independently observed block.
    grouped = defaultdict(list)

    for row in raw:
        key = (
            row["height"],
            row["block_hash"],
            row["parent_hash"],
        )
        grouped[key].append(row)

    sequences = []

    for key, items in grouped.items():
        height, block_hash, parent_hash = key

        # Ensure all rows closing against one block agree on target.
        states = {item["nonce_state"] for item in items}

        if len(states) != 1:
            raise RuntimeError(
                f"height {height}: inconsistent target states {states}"
            )

        state = next(iter(states))

        # Deduplicate BEFORE observations by notify fingerprint.
        by_fingerprint = {}

        for item in items:
            before = item["before"]
            fp = str(before["notify_fingerprint"])

            existing = by_fingerprint.get(fp)

            if existing is None:
                by_fingerprint[fp] = item
            else:
                # Duplicate across collectors is okay only if it describes
                # the same causal context.
                if (
                    existing["parent_hash"] != item["parent_hash"]
                    or existing["height"] != item["height"]
                ):
                    raise RuntimeError(
                        f"fingerprint collision: {fp}"
                    )

        unique = list(by_fingerprint.values())

        unique.sort(
            key=lambda item: (
                int(item["before"]["frozen_at"]),
                int(item["before"]["freeze_id"]),
                str(item["before"]["notify_fingerprint"]),
            )
        )

        steps = []

        for item in unique:
            before = item["before"]
            cognition = before["cognition"]

            steps.append(
                {
                    "source": item["source"],
                    "freeze_id": int(before["freeze_id"]),
                    "frozen_at": int(before["frozen_at"]),
                    "job_id": str(before["job_id"]),
                    "prevhash_canonical": str(
                        before["prevhash_canonical"]
                    ),
                    "pool_difficulty": float(
                        before["pool_difficulty"]
                    ),
                    "leader": cognition.get("leader"),
                    "leader_streak": int(
                        cognition["leader_streak"]
                    ),
                    "incoming": int(
                        cognition.get(
                            "incoming",
                            before["incoming"],
                        )
                    ),
                    "cognitive_cycles": int(
                        cognition["cognitive_cycles"]
                    ),
                    "recognition": float(
                        cognition["recognition"]
                    ),
                    "stability": float(
                        cognition["stability"]
                    ),
                    "confidence_internal": float(
                        cognition["confidence_internal"]
                    ),
                    "probabilities": [
                        float(v)
                        for v in cognition["probabilities"]
                    ],
                    "evidence": [
                        float(v)
                        for v in cognition["evidence"]
                    ],
                    "observations": [
                        int(v)
                        for v in cognition["observations"]
                    ],
                    "features": build_features(before),
                }
            )

        representative = unique[0]["outcome"]

        label = {
            "height": height,
            "hash": block_hash,
            "previousblockhash": parent_hash,
            "nonce": int(representative["nonce"]),
            "nonce_state": state,
            "version": int(representative["version"]),
            "bits": int(representative["bits"]),
            "difficulty": float(
                representative["difficulty"]
            ),
            "timestamp": int(representative["timestamp"]),
            "mediantime": int(representative["mediantime"]),
            "tx_count": int(representative["tx_count"]),
            "size": int(representative["size"]),
            "weight": int(representative["weight"]),
        }

        sequences.append(
            {
                "schema": "x1331-live08e2-sequence-v1",
                "sequence_id": (
                    f"btc-{height}-"
                    f"{block_hash[:16]}"
                ),
                "feature_dim": FEATURE_DIM,
                "length": len(steps),
                "parent_hash": parent_hash,
                "sources": sorted(
                    {item["source"] for item in unique}
                ),
                "steps": steps,
                "label": label,
            }
        )

    sequences.sort(
        key=lambda row: int(row["label"]["height"])
    )

    heights = [
        int(row["label"]["height"])
        for row in sequences
    ]

    if len(heights) != len(set(heights)):
        raise RuntimeError(
            "duplicate independent block outcomes remain"
        )

    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for row in sequences:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            f.write("\n")

    tmp.replace(OUTPUT)

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()

    manifest = {
        "schema": "x1331-live08e2-manifest-v1",
        "sources": [
            str(OLD),
            str(NEW),
        ],
        "source_rows": len(raw),
        "independent_outcomes": len(sequences),
        "heights": heights,
        "sequence_lengths": [
            row["length"] for row in sequences
        ],
        "feature_dim": FEATURE_DIM,
        "output": str(OUTPUT),
        "output_sha256": digest,
        "rules": {
            "before_only": True,
            "causal_relation": "DIRECT_SUCCESSOR",
            "unit_of_independence": "bitcoin_block",
            "cross_block_split": False,
            "notify_fingerprint_dedup": True,
        },
    }

    tmp_manifest = MANIFEST.with_suffix(
        MANIFEST.suffix + ".tmp"
    )

    with tmp_manifest.open("w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    tmp_manifest.replace(MANIFEST)

    print()
    print(f"source rows          = {len(raw)}")
    print(f"independent outcomes = {len(sequences)}")
    print(f"heights              = {heights}")
    print(
        "sequence lengths     = "
        f"{[row['length'] for row in sequences]}"
    )
    print(f"feature dimension    = {FEATURE_DIM}")
    print(f"output               = {OUTPUT}")
    print(f"output sha256        = {digest}")
    print()
    print("CAUSAL LINKAGE       = PASS")
    print("BEFORE-ONLY FEATURES = PASS")
    print("BLOCK GROUPING       = PASS")
    print("CROSS-BLOCK SPLIT    = ENFORCED")


if __name__ == "__main__":
    main()
