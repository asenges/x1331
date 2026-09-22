# X1331 LIVE-08F Prospective Evaluation Protocol

Frozen before observing the outcome of Bitcoin block 968067.

## Experiment

LIVE-08F is a frozen prospective shadow experiment.

Model SHA256:

c64d0c13acb9969ed25e2e114586ff77346de0115a7203179b00ff2a38242ecc

Training cutoff:

968063

Training outcomes:

968056 through 968063 inclusive.

## Target definition

LIVE-08F uses the LIVE-08E.2 target definition:

    nonce_state = (nonce >> 5) & 0b111

Therefore the eight X1331 states correspond to nonce bits 7..5:

    000
    001
    010
    011
    100
    101
    110
    111

This differs from LIVE-08E.0 / LIVE-08E.1, which used nonce bits 31..29.

The target definition MUST NOT change during this cohort.

## Pilot outcomes

Blocks 968065 and 968066 are classified as pilot outcomes because
their outcomes were inspected before this prospective cohort protocol
was frozen.

They may be reported separately but MUST NOT be included in the
primary prospective cohort statistics.

## Prospective cohort

First block:

968067

Cohort size:

32 consecutive independent Bitcoin block outcomes.

Expected cohort:

968067 through 968098 inclusive, subject to valid causal collection.

The frozen LIVE-08F model MUST NOT be retrained or tuned during this
cohort.

## Unit of evidence

One Bitcoin block is one independent outcome.

Multiple Stratum notifications / shadow freezes associated with the
same parent are NOT independent observations.

## Primary prediction rule

For each block, the primary prediction is the LAST durable LIVE-08F
SHADOW FREEZE recorded for its parent before the corresponding chain
outcome becomes known.

Earlier predictions from the same sequence are trajectory diagnostics
only.

They MUST NOT be counted as additional independent predictions.

## Primary metrics

For each independent block:

- actual nonce_state
- predicted state
- Top-1 hit or miss
- rank of actual state
- probability assigned to actual state
- log loss
- Brier score

Cohort summary:

- Top-1 accuracy
- mean log loss
- mean Brier score
- state prediction distribution
- actual state distribution

Uniform eight-state reference:

    P(state) = 0.125
    log loss = ln(8) = 2.079441542

## Model probability

LIVE-08F softmax output is a raw model probability.

It MUST NOT be interpreted as calibrated confidence or as the
probability that a Bitcoin hash will be valid.

## Safety / execution

LIVE-08F remains SHADOW ONLY.

SHA verification: OFF
Share submission: OFF
AUTO PLAY: OFF

The running LIVE-08D.1 collector and LIVE-08F shadow process must not
be modified as part of scoring.

## Freeze rule

No model parameter, feature definition, target definition, scoring
rule, cohort boundary, or primary metric may be changed based on
results observed inside this 32-block cohort.

Any later experiment using different rules must receive a new version
identifier and be evaluated separately.
