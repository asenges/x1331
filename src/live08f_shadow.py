#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


STATES = [
    "000", "001", "010", "011",
    "100", "101", "110", "111",
]

FEATURE_DIM = 39
HIDDEN_DIM = 32
NUM_LAYERS = 1

TRAINING_CUTOFF_HEIGHT = 968063
FIRST_ELIGIBLE_HEIGHT = 968064

EPOCHS = 300
LR = 0.003
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

SEED = 0x133108F0

SEQUENCES = Path("data/live08e2-sequences.jsonl")
CHECKPOINT = Path("data/live08d1-checkpoint.json")

MODEL_DIR = Path("data/live08f-model")
MODEL_FILE = MODEL_DIR / "model.pt"
NORMALIZATION_FILE = MODEL_DIR / "normalization.json"
MANIFEST_FILE = MODEL_DIR / "manifest.json"

SHADOW_FILE = Path("data/live08f-shadow.jsonl")
STATE_FILE = Path("data/live08f-state.json")


class RecurrentObserver(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        self.head = nn.Linear(
            HIDDEN_DIM,
            len(STATES),
        )

    def forward(self, x):
        _, hidden = self.gru(x)

        final_hidden = hidden[-1]

        return self.head(final_hidden)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def append_jsonl_durable(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    )

    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

    return rows


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


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

    incoming = cognition.get(
        "incoming",
        before.get("incoming"),
    )

    if incoming is None:
        raise RuntimeError("incoming missing")

    features = (
        [float(v) for v in probabilities]
        + [float(v) for v in evidence]
        + [float(v) for v in observations]
        + leader_one_hot
        + [
            float(cognition["recognition"]),
            float(cognition["stability"]),
            float(cognition["confidence_internal"]),
            float(cognition["leader_streak"]),
            float(cognition["cognitive_cycles"]),
            float(incoming),
            math.log1p(float(before["pool_difficulty"])),
        ]
    )

    if len(features) != FEATURE_DIM:
        raise RuntimeError(
            f"feature dimension {len(features)} != {FEATURE_DIM}"
        )

    return features


def load_training_sequences():
    rows = load_jsonl(SEQUENCES)

    selected = []

    for row in rows:
        height = int(row["label"]["height"])

        if height <= TRAINING_CUTOFF_HEIGHT:
            selected.append(row)

    selected.sort(
        key=lambda row: int(row["label"]["height"])
    )

    heights = [
        int(row["label"]["height"])
        for row in selected
    ]

    expected = list(
        range(
            heights[0],
            TRAINING_CUTOFF_HEIGHT + 1,
        )
    )

    if heights != expected:
        raise RuntimeError(
            "training heights are not contiguous: "
            f"{heights}"
        )

    if heights[-1] != TRAINING_CUTOFF_HEIGHT:
        raise RuntimeError(
            "training cutoff not present"
        )

    return selected


def compute_normalization(rows):
    all_steps = []

    for row in rows:
        for step in row["steps"]:
            features = step["features"]

            if len(features) != FEATURE_DIM:
                raise RuntimeError(
                    "bad training feature dimension"
                )

            all_steps.append(features)

    matrix = np.asarray(
        all_steps,
        dtype=np.float64,
    )

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)

    std[std < 1e-6] = 1.0

    return mean, std


def normalized_sequence(row, mean, std):
    matrix = np.asarray(
        [
            step["features"]
            for step in row["steps"]
        ],
        dtype=np.float32,
    )

    matrix = (
        matrix - mean.astype(np.float32)
    ) / std.astype(np.float32)

    return torch.tensor(
        matrix,
        dtype=torch.float32,
    ).unsqueeze(0)


def target_index(row):
    state = row["label"]["nonce_state"]

    if state not in STATES:
        raise RuntimeError(
            f"invalid nonce state: {state}"
        )

    return STATES.index(state)


def train_mode():
    print("X1331 LIVE-08F")
    print("FROZEN PROSPECTIVE SHADOW OBSERVER")
    print("MODE = TRAIN/FREEZE")
    print()

    if MODEL_FILE.exists() or MANIFEST_FILE.exists():
        raise SystemExit(
            "LIVE-08F model already frozen. "
            "Refusing to overwrite."
        )

    rows = load_training_sequences()

    if len(rows) != 8:
        raise RuntimeError(
            f"expected 8 training outcomes, got {len(rows)}"
        )

    heights = [
        int(row["label"]["height"])
        for row in rows
    ]

    print(f"training outcomes      = {len(rows)}")
    print(f"training heights       = {heights}")
    print(
        f"training cutoff height = "
        f"{TRAINING_CUTOFF_HEIGHT}"
    )
    print(
        f"first eligible height  = "
        f"{FIRST_ELIGIBLE_HEIGHT}"
    )

    source_sha = sha256_file(SEQUENCES)

    print(f"sequence source sha256 = {source_sha}")
    print()

    set_seed()

    mean, std = compute_normalization(rows)

    model = RecurrentObserver()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    last_loss = None

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        order = list(range(len(rows)))

        rng = random.Random(SEED + epoch)
        rng.shuffle(order)

        for idx in order:
            row = rows[idx]

            x = normalized_sequence(
                row,
                mean,
                std,
            )

            y = torch.tensor(
                [target_index(row)],
                dtype=torch.long,
            )

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            total_loss += float(loss.item())

        last_loss = total_loss / len(rows)

        if (
            epoch == 1
            or epoch % 50 == 0
            or epoch == EPOCHS
        ):
            print(
                f"epoch={epoch:03d} "
                f"mean_loss={last_loss:.9f}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_payload = {
        "schema": "x1331-live08f-model-v1",
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "states": STATES,
        "training_cutoff_height":
            TRAINING_CUTOFF_HEIGHT,
        "first_eligible_height":
            FIRST_ELIGIBLE_HEIGHT,
        "seed": SEED,
        "state_dict": model.state_dict(),
    }

    torch.save(
        model_payload,
        MODEL_FILE,
    )

    normalization = {
        "schema":
            "x1331-live08f-normalization-v1",
        "feature_dim": FEATURE_DIM,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    atomic_json(
        NORMALIZATION_FILE,
        normalization,
    )

    model_sha = sha256_file(MODEL_FILE)
    norm_sha = sha256_file(
        NORMALIZATION_FILE
    )

    manifest = {
        "schema":
            "x1331-live08f-manifest-v1",
        "created_at": int(time.time()),
        "mode":
            "FROZEN_PROSPECTIVE_SHADOW",
        "states": STATES,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "training_outcomes": len(rows),
        "training_heights": heights,
        "training_cutoff_height":
            TRAINING_CUTOFF_HEIGHT,
        "first_eligible_height":
            FIRST_ELIGIBLE_HEIGHT,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "seed": SEED,
        "final_training_loss": last_loss,
        "sequence_source":
            str(SEQUENCES),
        "sequence_source_sha256":
            source_sha,
        "model_file":
            str(MODEL_FILE),
        "model_sha256":
            model_sha,
        "normalization_file":
            str(NORMALIZATION_FILE),
        "normalization_sha256":
            norm_sha,
        "scientific_rules": {
            "model_probability_is_confidence":
                False,
            "prospective_only_after_cutoff":
                True,
            "retraining_during_evaluation":
                False,
            "sha256d":
                False,
            "verify":
                False,
            "submit":
                False,
            "auto_play":
                False,
        },
    }

    atomic_json(
        MANIFEST_FILE,
        manifest,
    )

    manifest_sha = sha256_file(
        MANIFEST_FILE
    )

    print()
    print("MODEL FROZEN")
    print(f"model         = {MODEL_FILE}")
    print(f"model sha256  = {model_sha}")
    print(
        f"normalization = "
        f"{NORMALIZATION_FILE}"
    )
    print(f"norm sha256   = {norm_sha}")
    print(
        f"manifest      = "
        f"{MANIFEST_FILE}"
    )
    print(f"manifest sha  = {manifest_sha}")
    print()
    print("SHA/VERIFY/SUBMIT/AUTO_PLAY = OFF")
    print(
        "MODEL PROBABILITY != "
        "CALIBRATED CONFIDENCE"
    )


def load_frozen_model():
    if not MODEL_FILE.exists():
        raise RuntimeError(
            "frozen model missing; run --train first"
        )

    if not NORMALIZATION_FILE.exists():
        raise RuntimeError(
            "normalization missing"
        )

    if not MANIFEST_FILE.exists():
        raise RuntimeError(
            "manifest missing"
        )

    manifest = json.loads(
        MANIFEST_FILE.read_text()
    )

    if (
        int(
            manifest[
                "training_cutoff_height"
            ]
        )
        != TRAINING_CUTOFF_HEIGHT
    ):
        raise RuntimeError(
            "manifest cutoff mismatch"
        )

    expected_model_sha = manifest[
        "model_sha256"
    ]

    actual_model_sha = sha256_file(
        MODEL_FILE
    )

    if actual_model_sha != expected_model_sha:
        raise RuntimeError(
            "MODEL HASH MISMATCH"
        )

    expected_norm_sha = manifest[
        "normalization_sha256"
    ]

    actual_norm_sha = sha256_file(
        NORMALIZATION_FILE
    )

    if actual_norm_sha != expected_norm_sha:
        raise RuntimeError(
            "NORMALIZATION HASH MISMATCH"
        )

    payload = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    model = RecurrentObserver()

    model.load_state_dict(
        payload["state_dict"]
    )

    model.eval()

    normalization = json.loads(
        NORMALIZATION_FILE.read_text()
    )

    mean = np.asarray(
        normalization["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        normalization["std"],
        dtype=np.float32,
    )

    return (
        model,
        mean,
        std,
        manifest,
        actual_model_sha,
    )


def load_shadowed_fingerprints():
    seen = set()

    if not SHADOW_FILE.exists():
        return seen

    with SHADOW_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            fp = row.get(
                "notify_fingerprint"
            )

            if fp:
                seen.add(str(fp))

    return seen


def entropy_normalized(probabilities):
    entropy = 0.0

    for p in probabilities:
        if p > 0.0:
            entropy -= p * math.log(p)

    return entropy / math.log(len(STATES))


def shadow_mode(once=False):
    print("X1331 LIVE-08F")
    print("FROZEN PROSPECTIVE SHADOW OBSERVER")
    print("MODE = SHADOW")
    print()

    (
        model,
        mean,
        std,
        manifest,
        model_sha,
    ) = load_frozen_model()

    print(
        "training cutoff = "
        f"{manifest['training_cutoff_height']}"
    )
    print(
        "first eligible  = "
        f"{manifest['first_eligible_height']}"
    )
    print(f"model sha256    = {model_sha}")
    print(
        "SHA/VERIFY/SUBMIT/AUTO_PLAY = OFF"
    )
    print()

    seen = load_shadowed_fingerprints()

    while True:
        try:
            checkpoint = json.loads(
                CHECKPOINT.read_text()
            )
        except Exception as exc:
            print(
                f"CHECKPOINT READ ERROR: {exc}"
            )

            if once:
                return

            time.sleep(1.0)
            continue

        pending = checkpoint.get(
            "pending",
            [],
        )

        # Group pending freezes by parent.
        groups = {}

        for before in pending:
            parent = str(
                before["prevhash_canonical"]
            )

            groups.setdefault(
                parent,
                [],
            ).append(before)

        emitted = 0

        for parent, freezes in groups.items():
            freezes.sort(
                key=lambda row: (
                    int(row["frozen_at"]),
                    int(row["freeze_id"]),
                )
            )

            # Every new freeze gets an immutable
            # prospective prediction using all
            # BEFORE cognition available through
            # that freeze for the same parent.
            sequence = []

            for before in freezes:
                fp = str(
                    before[
                        "notify_fingerprint"
                    ]
                )

                sequence.append(
                    build_features(before)
                )

                if fp in seen:
                    continue

                matrix = np.asarray(
                    sequence,
                    dtype=np.float32,
                )

                matrix = (
                    matrix - mean
                ) / std

                x = torch.tensor(
                    matrix,
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x)

                    probabilities = (
                        torch.softmax(
                            logits,
                            dim=1,
                        )[0]
                        .cpu()
                        .numpy()
                        .astype(float)
                        .tolist()
                    )

                ranking = sorted(
                    range(len(STATES)),
                    key=lambda i:
                        probabilities[i],
                    reverse=True,
                )

                top1 = ranking[0]
                top2 = ranking[1]

                p1 = float(
                    probabilities[top1]
                )

                p2 = float(
                    probabilities[top2]
                )

                record = {
                    "schema":
                        "x1331-live08f-shadow-v1",
                    "frozen_prediction_at":
                        int(time.time()),
                    "training_cutoff_height":
                        TRAINING_CUTOFF_HEIGHT,
                    "first_eligible_height":
                        FIRST_ELIGIBLE_HEIGHT,
                    "model_sha256":
                        model_sha,
                    "parent_hash":
                        parent,
                    "freeze_id":
                        int(
                            before["freeze_id"]
                        ),
                    "frozen_at":
                        int(
                            before["frozen_at"]
                        ),
                    "incoming":
                        int(
                            before["incoming"]
                        ),
                    "job_id":
                        str(
                            before["job_id"]
                        ),
                    "notify_fingerprint":
                        fp,
                    "sequence_length":
                        len(sequence),
                    "probabilities": {
                        state:
                            float(
                                probabilities[i]
                            )
                        for i, state
                        in enumerate(STATES)
                    },
                    "predicted_state":
                        STATES[top1],
                    "model_probability":
                        p1,
                    "second_state":
                        STATES[top2],
                    "second_probability":
                        p2,
                    "margin":
                        p1 - p2,
                    "normalized_entropy":
                        entropy_normalized(
                            probabilities
                        ),
                    "observer": {
                        "leader":
                            before[
                                "cognition"
                            ].get(
                                "leader"
                            ),
                        "recognition":
                            float(
                                before[
                                    "cognition"
                                ][
                                    "recognition"
                                ]
                            ),
                        "stability":
                            float(
                                before[
                                    "cognition"
                                ][
                                    "stability"
                                ]
                            ),
                        "confidence_internal":
                            float(
                                before[
                                    "cognition"
                                ][
                                    "confidence_internal"
                                ]
                            ),
                    },
                    "action":
                        "SHADOW_ONLY",
                    "sha256d":
                        False,
                    "verify":
                        False,
                    "submit":
                        False,
                    "auto_play":
                        False,
                }

                append_jsonl_durable(
                    SHADOW_FILE,
                    record,
                )

                seen.add(fp)
                emitted += 1

                print(
                    "SHADOW FREEZE | "
                    f"id={record['freeze_id']} "
                    f"parent={parent} "
                    f"seq={len(sequence)} "
                    f"pred="
                    f"{record['predicted_state']} "
                    f"p="
                    f"{record['model_probability']:.6f} "
                    f"margin="
                    f"{record['margin']:.6f} "
                    f"H="
                    f"{record['normalized_entropy']:.6f}"
                )

        state = {
            "schema":
                "x1331-live08f-state-v1",
            "updated_at":
                int(time.time()),
            "mode":
                "FROZEN_PROSPECTIVE_SHADOW",
            "training_cutoff_height":
                TRAINING_CUTOFF_HEIGHT,
            "first_eligible_height":
                FIRST_ELIGIBLE_HEIGHT,
            "model_sha256":
                model_sha,
            "shadow_records":
                len(seen),
            "pending_groups":
                len(groups),
            "pending_freezes":
                len(pending),
            "last_checkpoint_saved_at":
                checkpoint.get(
                    "saved_at"
                ),
            "sha256d":
                False,
            "verify":
                False,
            "submit":
                False,
            "auto_play":
                False,
        }

        atomic_json(
            STATE_FILE,
            state,
        )

        if emitted:
            print(
                f"STATE | shadow={len(seen)} "
                f"pending={len(pending)} "
                f"groups={len(groups)}"
            )

        if once:
            return

        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--train",
        action="store_true",
    )

    mode.add_argument(
        "--shadow",
        action="store_true",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="process current checkpoint once and exit",
    )

    args = parser.parse_args()

    if args.train:
        train_mode()
        return

    shadow_mode(
        once=args.once
    )


if __name__ == "__main__":
    main()
