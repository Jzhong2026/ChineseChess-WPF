#!/usr/bin/env python3
"""
V7 Two-Stage Policy Head design.

Problem with factored from/to head:
  - from_acc ~25% (okay-ish)
  - to_acc   ~8%  (very weak — the model cannot predict WHERE to move)
  - Combined Top-10 ~33% (weak policy prior for MCTS)

Root cause: predicting from and to independently ignores the STRONG correlation
between which piece moves and where it can go. E.g. a Horse at (2,1) can only
go to 8 specific squares; a Rook at (0,0) can go to 17 squares.
Independent prediction wastes model capacity.

V7 Two-Stage Policy Head:
  Stage 1 — from_logits [90]: which square the piece moves FROM.
  Stage 2 — to_logits_given_from [90, 90]: for each possible from square,
            predict where it can go. But we only need the conditional distribution
            P(to | from) = softmax(to_logits_given_from[from]) [to].

Implementation options:

Option A: Full [90, 90] to_logits matrix
  - Output: from_logits[90] + to_logits[90,90]
  - Combined: policy[from*90+to] = from_logits[from] + to_logits[from, to]
  - Params: 90*90 = 8,100 (still small)
  - Problem: to_logits[from, to] is only meaningful when (from,to) is a legal move.
    We need to mask illegal moves anyway, so this is fine.

Option B: Conditional to head (smarter, less params)
  - from_embedding[90, 64] → concatenated with board features → to_logits[90]
  - But this requires two forward passes (or a separate head per from).
  - Complex to implement in ONNX.

Option C: Keep factored but add FROM-TO correlation via attention
  - Add a small attention layer between from_features and to_features.
  - Complex.

→ Choose Option A: Full [90, 90] to_logits matrix.
  Easy to implement, easy to export to ONNX, easy to use in C#.
  The combined logit is still: policy[from*90+to] = from_logits[from] + to_logits[from, to]
  (This is mathematically equivalent to P(from,to) ∝ P(from) * P(to|from).)

Wait — this is exactly what the factored head ALREADY does!
  from_logits[from] + to_logits[to]  ← this assumes P(to) is independent of from.
  from_logits[from] + to_logits[from, to] ← this is P(from,to) ∝ P(from) * P(to|from).

So the fix is: change the to_head to be CONDITIONAL on from.
But in a single forward pass, we don't know which from the model will pick.

ALPHAZERO-STYLE FIX:
  In MCTS, when we evaluate a leaf node, we get from_logits[90] and to_logits[90,90].
  For each legal move (from,to), the policy prior is:
    P(from,to) = softmax(from_logits)[from] * softmax(to_logits[from,:])[to]
  But this is two softmax operations, which is expensive.

  Alternatively, use log-space:
    log_P(from,to) = from_logits[from] + to_logits[from, to] - log_Z
  where log_Z is the log-partition function (can be approximated).

Wait, this is exactly the same as the current factored head if to_logits is [90,90]!
The current to_logits[90] is actually to_logits_for_ALL_from, which is wrong.

CORRECT V7 DESIGN:
  Output 1: from_logits [90]
  Output 2: to_logits   [90, 90]   ← now conditional on from
  Combined: combined[from*90+to] = from_logits[from] + to_logits[from, to]
  
  Model architecture:
    - from_head: [CNN features] → from_logits[90]
    - to_head:   [CNN features] → to_logits[90, 90]
                  Specifically: a small network that takes CNN features and outputs [90, 90].
  
  ONNX export: 2 outputs (from_logits, to_logits, value_pred)
  C# inference: combined[from*90+to] = from_logits[from] + to_logits[from, to]

This should significantly improve to_accuracy because to_logits[from,to] can learn
that a Horse at (2,1) can only go to specific squares.
