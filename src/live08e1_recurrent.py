#!/usr/bin/env python3

import json
import math
import random
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# X1331 LIVE-08E.1
# RECURRENT SCHRODINGER OBSERVER
#
# Rules:
#   - BEFORE-only features
#   - one Bitcoin block = one independent outcome
#   - chronological walk-forward only
#   - normalization fitted on TRAIN only
#   - no random train/test split
#   - no SHA256d
#   - no VERIFY
#   - no SUBMIT
#   - no AUTO PLAY
# ============================================================

INPUT = Path("data/live08e-sequences.jsonl")
OUTPUT = Path("data/live08e1-results.json")
MODEL_DIR = Path("data/live08e1-models")

STATES = [
    "000", "001", "010", "011",
    "100", "101", "110", "111",
]

FEATURE_DIM = 39
HIDDEN_DIM = 32
NUM_LAYERS = 1

MIN_TRAIN_OUTCOMES = 3

EPOCHS = 300
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

SEED = 0x133108E1

DEVICE = torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def state_index(state: str) -> int:
    if state not in STATES:
        raise ValueError(f"invalid X1331 state: {state}")
    return STATES.index(state)


def load_sequences(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            if row.get("schema") != "x1331-live08e-sequence-v1":
                raise ValueError(
                    f"line {line_no}: unexpected schema "
                    f"{row.get('schema')!r}"
                )

            if int(row.get("feature_dim", -1)) != FEATURE_DIM:
                raise ValueError(
                    f"line {line_no}: feature_dim="
                    f"{row.get('feature_dim')} expected={FEATURE_DIM}"
                )

            label = row.get("label", {})
            state = label.get("nonce_state")

            if state not in STATES:
                raise ValueError(
                    f"line {line_no}: invalid/missing nonce_state={state!r}"
                )

            steps = row.get("steps", [])

            if not steps:
                raise ValueError(f"line {line_no}: empty sequence")

            features = []

            for step_no, step in enumerate(steps, 1):
                x = step.get("features")

                if not isinstance(x, list) or len(x) != FEATURE_DIM:
                    raise ValueError(
                        f"line {line_no} step {step_no}: "
                        f"expected {FEATURE_DIM} features"
                    )

                x = [float(v) for v in x]

                if not all(math.isfinite(v) for v in x):
                    raise ValueError(
                        f"line {line_no} step {step_no}: "
                        "non-finite feature"
                    )

                features.append(x)

            declared_length = int(row.get("length", len(features)))

            if declared_length != len(features):
                raise ValueError(
                    f"line {line_no}: declared length={declared_length} "
                    f"actual={len(features)}"
                )

            height = int(label["height"])

            rows.append(
                {
                    "sequence_id": row["sequence_id"],
                    "parent_hash": row["parent_hash"],
                    "height": height,
                    "state": state,
                    "target": state_index(state),
                    "features": np.asarray(features, dtype=np.float32),
                }
            )

    rows.sort(key=lambda r: r["height"])

    heights = [r["height"] for r in rows]

    if len(set(heights)) != len(heights):
        raise ValueError("duplicate outcome heights detected")

    return rows


class TrainNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sequences):
        all_steps = np.concatenate(
            [row["features"] for row in sequences],
            axis=0,
        ).astype(np.float64)

        self.mean = all_steps.mean(axis=0)
        self.std = all_steps.std(axis=0)

        # Avoid exploding constant / nearly constant dimensions.
        self.std[self.std < 1e-6] = 1.0

    def transform(self, x):
        y = (x.astype(np.float64) - self.mean) / self.std
        return y.astype(np.float32)

    def as_json(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class SchrodingerGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=FEATURE_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )

        self.head = nn.Linear(HIDDEN_DIM, len(STATES))

    def forward(self, x):
        _, h = self.gru(x)

        last_hidden = h[-1]

        logits = self.head(last_hidden)

        return logits


def tensor_for_sequence(row, normalizer):
    x = normalizer.transform(row["features"])
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)


def train_fold(train_rows, fold_seed):
    seed_everything(fold_seed)

    normalizer = TrainNormalizer()
    normalizer.fit(train_rows)

    model = SchrodingerGRU().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    final_loss = None

    for _epoch in range(EPOCHS):
        order = list(range(len(train_rows)))

        # Deterministic because seed_everything() was called for this fold.
        random.shuffle(order)

        epoch_loss = 0.0

        for i in order:
            row = train_rows[i]

            x = tensor_for_sequence(row, normalizer)
            target = torch.tensor(
                [row["target"]],
                dtype=torch.long,
                device=DEVICE,
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            epoch_loss += float(loss.item())

        final_loss = epoch_loss / len(train_rows)

    return model, normalizer, final_loss


@torch.no_grad()
def predict(model, normalizer, row):
    model.eval()

    x = tensor_for_sequence(row, normalizer)

    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)[0]

    return probabilities.cpu().numpy().astype(np.float64)


def actual_rank(probabilities, actual_index):
    order = np.argsort(-probabilities)

    for rank, idx in enumerate(order, 1):
        if int(idx) == actual_index:
            return rank

    raise RuntimeError("actual state disappeared from ranking")


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def save_model(path, model, normalizer, train_rows, test_row):
    payload = {
        "schema": "x1331-live08e1-model-v1",
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "states": STATES,
        "train_heights": [r["height"] for r in train_rows],
        "test_height": test_row["height"],
        "normalizer": normalizer.as_json(),
        "model_state_dict": model.state_dict(),
    }

    torch.save(payload, path)


def main():
    print("X1331 LIVE-08E.1")
    print("RECURRENT SCHRODINGER OBSERVER")
    print(
        "BEFORE-ONLY | WALK-FORWARD | "
        "AUTO PLAY=OFF | VERIFY=OFF | SUBMIT=OFF"
    )
    print()

    if not INPUT.exists():
        raise SystemExit(
            f"missing {INPUT}; run src/live08e_sequence_builder.py first"
        )

    rows = load_sequences(INPUT)

    print(f"source               = {INPUT}")
    print(f"source sha256        = {sha256_file(INPUT)}")
    print(f"independent outcomes = {len(rows)}")
    print(f"feature dimension    = {FEATURE_DIM}")
    print(f"sequence lengths     = {[len(r['features']) for r in rows]}")
    print(f"device               = {DEVICE}")
    print(f"hidden dimension     = {HIDDEN_DIM}")
    print(f"epochs/fold          = {EPOCHS}")
    print()

    if len(rows) <= MIN_TRAIN_OUTCOMES:
        raise SystemExit(
            "not enough independent outcomes for first walk-forward test"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    uniform_probability = 1.0 / len(STATES)
    uniform_log_loss = -math.log(uniform_probability)

    for test_pos in range(MIN_TRAIN_OUTCOMES, len(rows)):
        train_rows = rows[:test_pos]
        test_row = rows[test_pos]

        fold_number = test_pos - MIN_TRAIN_OUTCOMES + 1
        fold_seed = SEED + fold_number

        model, normalizer, train_loss = train_fold(
            train_rows,
            fold_seed,
        )

        probabilities = predict(
            model,
            normalizer,
            test_row,
        )

        actual_idx = test_row["target"]
        predicted_idx = int(np.argmax(probabilities))

        predicted_state = STATES[predicted_idx]
        actual_state = test_row["state"]

        p_actual = max(float(probabilities[actual_idx]), 1e-12)
        log_loss = -math.log(p_actual)
        rank = actual_rank(probabilities, actual_idx)

        top2 = set(np.argsort(-probabilities)[:2].tolist())
        top4 = set(np.argsort(-probabilities)[:4].tolist())

        result = {
            "fold": fold_number,
            "train_heights": [r["height"] for r in train_rows],
            "test_height": test_row["height"],
            "sequence_id": test_row["sequence_id"],
            "sequence_length": len(test_row["features"]),
            "actual_state": actual_state,
            "predicted_state": predicted_state,
            "actual_rank": rank,
            "actual_probability": p_actual,
            "probabilities": {
                state: float(probabilities[i])
                for i, state in enumerate(STATES)
            },
            "log_loss": log_loss,
            "uniform_log_loss": uniform_log_loss,
            "top1_hit": predicted_idx == actual_idx,
            "top2_hit": actual_idx in top2,
            "top4_hit": actual_idx in top4,
            "final_train_loss": train_loss,
            "fold_seed": fold_seed,
        }

        results.append(result)

        print("=" * 72)
        print(
            f"FOLD {fold_number} | "
            f"train={len(train_rows)} outcomes -> "
            f"test height={test_row['height']}"
        )
        print(
            f"BEFORE sequence length = "
            f"{len(test_row['features'])}"
        )
        print()

        for i, state in enumerate(STATES):
            marker = ""
            if state == actual_state:
                marker += " ACTUAL"
            if state == predicted_state:
                marker += " TOP"

            print(
                f"P({state}) = "
                f"{float(probabilities[i]):.9f}{marker}"
            )

        print()
        print(f"predicted state = {predicted_state}")
        print(f"actual state    = {actual_state}")
        print(f"rank(actual)    = {rank}/8")
        print(f"P(actual)       = {p_actual:.9f}")
        print(f"log loss        = {log_loss:.9f}")
        print(f"uniform loss    = {uniform_log_loss:.9f}")
        print(
            f"top1/top2/top4  = "
            f"{result['top1_hit']}/"
            f"{result['top2_hit']}/"
            f"{result['top4_hit']}"
        )
        print(f"train loss      = {train_loss:.9f}")

        model_path = MODEL_DIR / (
            f"fold-{fold_number:03d}-"
            f"test-{test_row['height']}.pt"
        )

        save_model(
            model_path,
            model,
            normalizer,
            train_rows,
            test_row,
        )

        print(f"model           = {model_path}")

    mean_loss = float(
        np.mean([r["log_loss"] for r in results])
    )

    mean_rank = float(
        np.mean([r["actual_rank"] for r in results])
    )

    top1_hits = sum(int(r["top1_hit"]) for r in results)
    top2_hits = sum(int(r["top2_hit"]) for r in results)
    top4_hits = sum(int(r["top4_hit"]) for r in results)

    actual_probabilities = [
        r["actual_probability"] for r in results
    ]

    geometric_probability = math.exp(
        float(np.mean(np.log(np.maximum(actual_probabilities, 1e-12))))
    )

    summary = {
        "schema": "x1331-live08e1-results-v1",
        "source": str(INPUT),
        "source_sha256": sha256_file(INPUT),
        "independent_outcomes": len(rows),
        "evaluated_outcomes": len(results),
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs_per_fold": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "uniform_probability": uniform_probability,
        "uniform_log_loss": uniform_log_loss,
        "mean_log_loss": mean_loss,
        "mean_actual_rank": mean_rank,
        "geometric_mean_actual_probability": geometric_probability,
        "top1_hits": top1_hits,
        "top2_hits": top2_hits,
        "top4_hits": top4_hits,
        "folds": results,
        "scientific_status": (
            "PIPELINE TEST ONLY - STATISTICAL ADEQUACY UNPROVEN"
        ),
    }

    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    tmp.replace(OUTPUT)

    print()
    print("=" * 72)
    print("LIVE-08E.1 WALK-FORWARD SUMMARY")
    print("=" * 72)
    print(f"evaluated outcomes   = {len(results)}")
    print(f"mean log loss        = {mean_loss:.9f}")
    print(f"uniform log loss     = {uniform_log_loss:.9f}")
    print(f"mean actual rank     = {mean_rank:.6f}")
    print(
        f"top1 hits            = "
        f"{top1_hits}/{len(results)}"
    )
    print(
        f"top2 hits            = "
        f"{top2_hits}/{len(results)}"
    )
    print(
        f"top4 hits            = "
        f"{top4_hits}/{len(results)}"
    )
    print(
        f"geo mean P(actual)   = "
        f"{geometric_probability:.9f}"
    )
    print(f"uniform P(state)     = {uniform_probability:.9f}")
    print(f"results              = {OUTPUT}")
    print()
    print(
        "STATUS = PIPELINE TEST ONLY; "
        "STATISTICAL ADEQUACY UNPROVEN"
    )


if __name__ == "__main__":
    main()
