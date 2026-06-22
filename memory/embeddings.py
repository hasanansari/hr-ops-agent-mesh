"""Turns an anomaly's context into a fixed-length numeric vector for
similarity search in the episodic memory store.

Deliberately NOT a sentence-transformer / NLP embedding model. Three
reasons: (1) this environment has no guaranteed network access to download
a pretrained model on first use, so relying on one would be a real
reliability risk; (2) a pretrained text embedding's notion of "similar" is
a black box -- it can't be explained in plain English the way every other
scoring method in this project can; (3) what actually makes two HR
incidents "similar" for this system's purposes isn't prose similarity, it's
similarity in the same structured signals Sections B/C/E already compute
(anomaly type, confidence/severity, direction of the discrepancy, employee
tenure). A small, hand-built feature vector captures exactly that, stays
fully explainable, and needs no external dependency beyond numpy.

Layout (6 dimensions):
    [0] is_payroll_outlier
    [1] is_leave_abuse
    [2] is_compliance_violation
    [3] confidence            -- already a 0-1 severity signal by Section
                                  B's own design (z-score scaled for
                                  payroll, blended ratio for leave,
                                  severity-based for compliance), reused
                                  here rather than inventing a duplicate
                                  "how bad is this" feature
    [4] payroll_direction     -- sign of the z-score for payroll outliers
                                  only (+1 overpaid, -1 underpaid, 0 for
                                  non-payroll types); "underpaid by a lot"
                                  and "overpaid by a lot" are different
                                  real-world situations even at equal
                                  confidence, so this keeps them apart
    [5] tenure_normalized     -- employee tenure_months / 60, capped at 1.0;
                                  ties memory similarity to the same
                                  probation distinction Section E's rules
                                  care about (a probation-period incident
                                  is a meaningfully different situation)
"""

from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 6


def embed_incident(anomaly_type: str, confidence: float, evidence: dict, employee: dict | None) -> list[float]:
    is_payroll = 1.0 if anomaly_type == "payroll_outlier" else 0.0
    is_leave = 1.0 if anomaly_type == "leave_abuse" else 0.0
    is_compliance = 1.0 if anomaly_type == "compliance_violation" else 0.0

    payroll_direction = 0.0
    if is_payroll and "z_score" in evidence:
        z = evidence["z_score"]
        payroll_direction = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)

    tenure_months = (employee or {}).get("tenure_months")
    tenure_normalized = min(1.0, tenure_months / 60.0) if tenure_months is not None else 0.0

    vector = np.array(
        [is_payroll, is_leave, is_compliance, float(confidence), payroll_direction, tenure_normalized]
    )
    return vector.tolist()
