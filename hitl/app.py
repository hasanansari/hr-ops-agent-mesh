import json
import sys
from pathlib import Path

# Streamlit only puts this script's own folder (hitl/) on sys.path, not the
# project root -- so sibling packages like `agents` can't be found unless we
# add the root ourselves. Has to happen before the agents/hitl imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from agents.anomaly_models import RecommendedAction
from hitl import store
from hitl.models import HITLDecisionType, edit_distance

st.set_page_config(page_title="HR Ops -- HITL Review Queue", layout="wide")
st.title("HR Ops -- Human-in-the-Loop Review Queue")
st.caption(
    "High-confidence anomalies waiting for approval before the Action Agent executes. "
    "Items left untouched will time out into a safe fallback action."
)

ANOMALY_ICONS = {
    "payroll_outlier": "\U0001f4b0",
    "leave_abuse": "\U0001f334",
    "compliance_violation": "⚠️",
}


def _evidence_summary(anomaly_type: str, evidence: dict) -> str:
    """One human-readable sentence per anomaly type, stating the raw observed
    facts only -- derived statistics (z-score, clustering ratio, severity %)
    are deliberately left out here and shown only behind the "raw evidence"
    checkbox, so this line stays plain English instead of mixing in the math
    that produced the score.
    """
    if anomaly_type == "payroll_outlier":
        direction = "underpaid" if evidence["z_score"] < 0 else "overpaid"
        # a pair of bare $ signs is interpreted as inline LaTeX by st.markdown
        # -- \$ escapes it so the currency amounts render as plain text.
        return (
            f"{evidence['department']}, Level {evidence['level']} -- salary "
            f"\\${evidence['salary']:,.2f} vs a peer average of \\${evidence['cohort_mean']:,.2f} "
            f"-- **{direction} relative to peers**."
        )
    if anomaly_type == "leave_abuse":
        return (
            f"Took **{evidence['leave_taken_qtr']} days** of leave this quarter "
            f"(policy limit: {evidence['policy_limit_qtr']} days), including **{evidence['long_weekend_count']}** "
            f"single Mondays/Fridays -- a pattern of extending weekends."
        )
    if anomaly_type == "compliance_violation":
        if evidence.get("violation") == "missing_mandatory_training":
            return "Has **not completed** mandatory training."
        if evidence.get("violation") == "overtime_cap_breach":
            return (
                f"Worked **{evidence['overtime_hours_week']} hrs** of overtime this week, "
                f"against a **{evidence['cap']} hr** cap."
            )
    return json.dumps(evidence)


def _action_label(action: str) -> str:
    """Display label for a raw action value like 'escalate-to-HR' -- the
    underlying value submitted to the store stays unchanged, this is purely
    cosmetic. .title() alone would turn 'HR' into 'Hr', hence the fixup.
    """
    return action.replace("-", " ").title().replace("Hr", "HR")


reviewer = st.sidebar.text_input("Your name (recorded with each decision)", value="anonymous")

pending = store.list_pending()

if not pending:
    st.info("No pending reviews right now. Run a scan from the agent graph to populate this queue.")
else:
    st.write(f"**{len(pending)} item(s) pending review**")
    for item in pending:
        evidence = json.loads(item["evidence_json"])
        icon = ANOMALY_ICONS.get(item["anomaly_type"], "❓")
        label = item["anomaly_type"].replace("_", " ").title()
        title = (
            f"{icon} {item['employee_id']} -- {label} -- "
            f"confidence {item['confidence']:.2f} -- proposed: {_action_label(item['proposed_action'])}"
        )
        with st.expander(title):
            st.markdown(_evidence_summary(item["anomaly_type"], evidence))
            if st.checkbox("Show raw evidence (exact numbers)", key=f"raw_{item['anomaly_id']}"):
                st.json(evidence)
            st.caption(f"anomaly_id: {item['anomaly_id']}")

            decision = st.radio(
                "Decision",
                ["Approve", "Reject", "Modify"],
                key=f"decision_{item['anomaly_id']}",
                horizontal=True,
            )

            if decision == "Approve":
                if st.button("Submit approval", key=f"submit_{item['anomaly_id']}"):
                    store.submit_decision(
                        anomaly_id=item["anomaly_id"],
                        decision=HITLDecisionType.APPROVE.value,
                        final_action=item["proposed_action"],
                        reviewer=reviewer,
                    )
                    st.rerun()

            elif decision == "Reject":
                reason = st.text_input("Rejection reason (required)", key=f"reason_{item['anomaly_id']}")
                if st.button("Submit rejection", key=f"submit_{item['anomaly_id']}"):
                    if not reason.strip():
                        st.error("A rejection reason is required.")
                    else:
                        store.submit_decision(
                            anomaly_id=item["anomaly_id"],
                            decision=HITLDecisionType.REJECT.value,
                            final_action=RecommendedAction.NO_ACTION.value,
                            rejection_reason=reason.strip(),
                            reviewer=reviewer,
                        )
                        st.rerun()

            else:  # Modify
                options = [a.value for a in RecommendedAction]
                new_action = st.selectbox(
                    "New action",
                    options,
                    index=options.index(item["proposed_action"]),
                    format_func=_action_label,
                    key=f"modify_{item['anomaly_id']}",
                )
                if st.button("Submit modification", key=f"submit_{item['anomaly_id']}"):
                    dist = edit_distance(RecommendedAction(item["proposed_action"]), RecommendedAction(new_action))
                    store.submit_decision(
                        anomaly_id=item["anomaly_id"],
                        decision=HITLDecisionType.MODIFY.value,
                        final_action=new_action,
                        edit_distance_value=dist,
                        reviewer=reviewer,
                    )
                    st.rerun()

st.divider()
st.subheader("Decision history")
history = [d for d in store.list_all() if d["status"] != "pending"]

if not history:
    st.caption("No decisions recorded yet.")
else:
    df = pd.DataFrame(history)

    OUTCOME_LABELS = {
        "approve": "Approved",
        "reject": "Rejected",
        "modify": "Modified",
    }

    def _outcome(row: pd.Series) -> str:
        if row["is_timeout_fallback"]:
            return "Timed out"
        return OUTCOME_LABELS.get(row["human_decision"], row["human_decision"])

    def _detail(row: pd.Series) -> str:
        if row["is_timeout_fallback"]:
            return "No response within review window"
        if row["human_decision"] == "reject":
            return row["rejection_reason"] or ""
        if row["human_decision"] == "modify":
            return f"moved {int(row['edit_distance'])} step(s) from proposed"
        return ""

    def _decided_by(row: pd.Series) -> str:
        return "Timeout (auto)" if row["is_timeout_fallback"] else (row["reviewer"] or "anonymous")

    display_df = pd.DataFrame(
        {
            "Employee": df["employee_id"],
            "Anomaly": df["anomaly_type"].str.replace("_", " ").str.title(),
            "Proposed": df["proposed_action"],
            "Confidence": df["confidence"],
            "Outcome": df.apply(_outcome, axis=1),
            "Final Action": df["final_action"],
            "Detail": df.apply(_detail, axis=1),
            "Decided By": df.apply(_decided_by, axis=1),
            "When": pd.to_datetime(df["decided_at"]).dt.strftime("%b %d, %H:%M:%S"),
        }
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
