#!/usr/bin/env python3
"""
Support ticket triage pipeline.

Usage:
    python code/main.py

Reads:  support_tickets/support_tickets.csv
Writes: support_tickets/output.csv

Set TEST_MODE = True to process 10 random samples instead of the full dataset.
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retriever import Retriever
from agent import process_ticket

# ── Configuration ────────────────────────────────────────────────────────────

TEST_MODE = False  # True → randomly sample 10 rows; False → full 57-ticket run

REPO_ROOT = Path(__file__).parent.parent
INPUT_CSV  = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification",
]

_SCORE_THRESHOLD = 0.05   # below this → skip API, escalate as low-confidence

# ── Rule-based keyword groups ─────────────────────────────────────────────────
# Each group bypasses the API entirely.  Ordered highest-priority first.

_OUTAGE_KEYWORDS = [
    "site is down", "site down", "all pages down", "pages are accessible",
    "not working at all", "stopped working completely", "all requests are failing",
    "none of the submissions", "is down",
]
_IDENTITY_THEFT_KEYWORDS = [
    "identity theft", "identity stolen", "identity has been stolen",
    "identity was stolen", "stolen my identity",
]
_INVALID_KEYWORDS = [
    "delete all files", "give me the code to delete", "rm -rf",
    "règles internes", "show your internal rules", "display all the rules",
    "what is the name of the actor", "who plays iron man",
]
_BUG_BOUNTY_KEYWORDS = [
    "bug bounty", "security vulnerability", "found a vulnerability",
    "found a major security", "security flaw", "responsible disclosure",
]
_SCORE_DISPUTE_KEYWORDS = [
    "increase my score", "review my test", "review my answers",
    "change my score", "increase my result", "score dispute",
    "recruiter rejected", "rejected me",
]
_BILLING_ESCALATE_KEYWORDS = [
    "refund asap", "refund immediately", "refund today", "give me the refund",
    "make visa refund", "want a refund", "need a refund now",
]
_PAYMENT_FAIL_KEYWORDS = [
    "payment failed", "transaction failed",
    "payment not going through", "card declined",
]
_LOGIN_KEYWORDS = [
    "can't login", "cannot login", "unable to login",
    "can't log in", "cannot log in", "unable to log in",
    "account locked",
]
_PASSWORD_KEYWORDS = [
    "forgot password", "forgot my password", "reset password",
    "reset my password", "password reset", "lost my password",
]
_SUBSCRIPTION_KEYWORDS = [
    "pause subscription", "cancel subscription",
    "pause our subscription", "cancel our subscription",
    "cancel plan", "pause plan",
]
_HOWTO_KEYWORDS   = ["how do i", "how to", "where can i"]
_HOWTO_EXCLUSIONS = [
    "dispute", "card", "payment", "score", "refund",
    "visa", "charge", "money", "fraud",
]

# ── Canned results ────────────────────────────────────────────────────────────

_OUTAGE_RESULT = {
    "status": "escalated", "product_area": "general_support",
    "response": "Escalate to a human",
    "justification": "Service outage detected; requires human investigation.",
    "request_type": "bug",
}
_IDENTITY_ESCALATE_RESULT = {
    "status": "escalated", "product_area": "general_support",
    "response": "Escalate to a human",
    "justification": "Identity theft is a serious security incident requiring immediate human action.",
    "request_type": "product_issue",
}
_INVALID_RESULT = {
    "status": "replied", "product_area": "conversation_management",
    "response": "I'm sorry, this request is outside the scope of my support capabilities.",
    "justification": "Request is harmful, malicious, or completely off-topic.",
    "request_type": "invalid",
}
_BUG_BOUNTY_RESULT = {
    "status": "replied", "product_area": "safeguards",
    "response": (
        "Thank you for reporting this security concern. Please submit your finding "
        "through the official responsible disclosure or bug bounty program for the "
        "relevant product. Our security team will review and follow up on your report."
    ),
    "justification": "Security vulnerability report; directed to responsible disclosure process.",
    "request_type": "bug",
}
_SCORE_DISPUTE_RESULT = {
    "status": "replied", "product_area": "screen",
    "response": (
        "HackerRank records and grades submissions automatically and accurately. "
        "The platform cannot alter assessment scores or influence recruiter hiring "
        "decisions. For questions about the outcome of your assessment, please "
        "contact the recruiting company directly."
    ),
    "justification": "Score change or recruiter decision override is outside platform scope.",
    "request_type": "invalid",
}
_BILLING_ESCALATE_RESULT = {
    "status": "escalated", "product_area": "billing",
    "response": "Escalate to a human",
    "justification": "Urgent billing or refund dispute requires human review and action.",
    "request_type": "product_issue",
}
_PAYMENT_FAIL_RESULT = {
    "status": "replied", "product_area": "billing",
    "response": (
        "If your payment failed, please try the following: verify your card details "
        "are correct, ensure sufficient funds are available, check with your bank for "
        "any blocks on the transaction, then retry after a few minutes. If the issue "
        "persists, contact your bank or try an alternative payment method."
    ),
    "justification": "Standard payment failure troubleshooting; generic guidance provided.",
    "request_type": "product_issue",
}
_LOGIN_RESULT = {
    "status": "replied", "product_area": "account_management",
    "response": (
        "If you're having trouble logging in, try resetting your password using the "
        "'Forgot Password' link on the login page. If your account is locked, wait a "
        "few minutes before retrying. For enterprise accounts with admin-managed access, "
        "please contact your workspace administrator."
    ),
    "justification": "Standard login/access issue; generic guidance provided.",
    "request_type": "product_issue",
}
_PASSWORD_RESULT = {
    "status": "replied", "product_area": "account_management",
    "response": (
        "To reset your password, click 'Forgot Password' on the login page and follow "
        "the instructions sent to your registered email address. If the email doesn't "
        "arrive within a few minutes, check your spam folder."
    ),
    "justification": "Standard password reset request; generic guidance provided.",
    "request_type": "product_issue",
}
_SUBSCRIPTION_RESULT = {
    "status": "replied", "product_area": "billing",
    "response": (
        "To manage your subscription, log in to your account and navigate to the "
        "billing or account settings section. From there you can view, modify, pause, "
        "or cancel your plan. For enterprise or team subscriptions, please contact "
        "your account manager."
    ),
    "justification": "Subscription management request; generic guidance provided.",
    "request_type": "product_issue",
}
_HOWTO_RESULT = {
    "status": "replied", "product_area": "general_support",
    "response": (
        "Please visit the official support documentation for your product — "
        "HackerRank (support.hackerrank.com), Claude (support.claude.com), or "
        "Visa (visa.co.in/support) — for step-by-step guidance. If you need "
        "further help, provide more detail about your specific question."
    ),
    "justification": "Generic how-to query; directed to official support documentation.",
    "request_type": "product_issue",
}
_INSUFFICIENT_RESULT = {
    "status": "escalated", "product_area": "general_support",
    "response": "Escalate to a human",
    "justification": "Insufficient corpus evidence to answer safely.",
    "request_type": "product_issue",
}

# ── Rule engine ───────────────────────────────────────────────────────────────

def _quick_check(issue: str, subject: str) -> dict | None:
    text = (issue + " " + subject).lower()

    # Priority 1 — escalations (must run before any reply rule)
    if any(k in text for k in _OUTAGE_KEYWORDS):
        return _OUTAGE_RESULT
    if any(k in text for k in _IDENTITY_THEFT_KEYWORDS):
        return _IDENTITY_ESCALATE_RESULT

    # Priority 2 — invalid / harmful
    if any(k in text for k in _INVALID_KEYWORDS):
        return _INVALID_RESULT

    # Priority 3 — specialised reply rules (high confidence)
    if any(k in text for k in _BUG_BOUNTY_KEYWORDS):
        return _BUG_BOUNTY_RESULT
    if any(k in text for k in _SCORE_DISPUTE_KEYWORDS):
        return _SCORE_DISPUTE_RESULT

    # Priority 4 — billing
    if any(k in text for k in _BILLING_ESCALATE_KEYWORDS):
        return _BILLING_ESCALATE_RESULT
    if any(k in text for k in _PAYMENT_FAIL_KEYWORDS):
        return _PAYMENT_FAIL_RESULT

    # Priority 5 — account / auth
    if any(k in text for k in _LOGIN_KEYWORDS):
        return _LOGIN_RESULT
    if any(k in text for k in _PASSWORD_KEYWORDS):
        return _PASSWORD_RESULT

    # Priority 6 — plan management
    if any(k in text for k in _SUBSCRIPTION_KEYWORDS):
        return _SUBSCRIPTION_RESULT

    # Priority 7 — generic how-to (only if no financial / scoring topic present)
    if (any(k in text for k in _HOWTO_KEYWORDS)
            and not any(e in text for e in _HOWTO_EXCLUSIONS)):
        return _HOWTO_RESULT

    return None


# ── Company helpers ───────────────────────────────────────────────────────────

_COMPANY_NORM = {
    "hackerrank": "hackerrank", "claude": "claude", "visa": "visa",
    "none": None, "": None,
}

def _normalize_company(raw: str) -> str | None:
    return _COMPANY_NORM.get(raw.strip().lower())

def _infer_company(issue: str, subject: str) -> str | None:
    text = (issue + " " + subject).lower()
    if any(k in text for k in ("hackerrank", "test", "assessment", "interview", "candidate", "screen")):
        return "hackerrank"
    if any(k in text for k in ("claude", "anthropic", "bedrock", "conversation")):
        return "claude"
    if any(k in text for k in ("visa", "card", "payment", "merchant", "refund", "travel cheque")):
        return "visa"
    return None

def _pick_model(issue: str) -> str:
    """Use Sonnet for long/complex issues; Haiku otherwise."""
    if len(issue) > 200:
        return "claude-sonnet-4-6"
    return "claude-haiku-4-5-20251001"

# ── Cache ─────────────────────────────────────────────────────────────────────

_result_cache: dict[tuple[str, str], dict] = {}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading corpus and building TF-IDF index… (this takes a moment)")
    retriever = Retriever()
    print("Index ready.\n")

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    if TEST_MODE:
        rows = random.sample(all_rows, min(10, len(all_rows)))
        print(f"TEST MODE — processing {len(rows)} random samples out of {len(all_rows)} total\n")
    else:
        rows = all_rows
        print(f"Processing {len(rows)} tickets → {OUTPUT_CSV}\n")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            issue      = row.get("Issue", "").strip()
            subject    = row.get("Subject", "").strip()
            company_raw = row.get("Company", "").strip()

            company = _normalize_company(company_raw)
            if company is None:
                company = _infer_company(issue, subject)

            cache_key = (issue, subject)

            result = _quick_check(issue, subject)
            if result is not None:
                label = "RULE  "
            elif cache_key in _result_cache:
                result = _result_cache[cache_key]
                label  = "CACHE "
            else:
                query      = f"{subject} {issue}".strip()
                docs       = retriever.retrieve(query, company, top_k=2)
                best_score = max((d["score"] for d in docs), default=0.0)

                if best_score < _SCORE_THRESHOLD:
                    result = _INSUFFICIENT_RESULT
                    label  = "LOW   "
                else:
                    model  = _pick_model(issue)
                    result = process_ticket(issue, subject, company_raw, docs, model=model)
                    label  = "SONNET" if "sonnet" in model else "HAIKU "

                _result_cache[cache_key] = result

            writer.writerow({
                "issue":        issue,
                "subject":      subject,
                "company":      company_raw,
                "response":     result.get("response", ""),
                "product_area": result.get("product_area", ""),
                "status":       result.get("status", "escalated"),
                "request_type": result.get("request_type", "product_issue"),
                "justification":result.get("justification", ""),
            })
            out_f.flush()

            icon = "✓" if result.get("status") == "replied" else "↑"
            print(
                f"[{i:02d}/{len(rows)}] [{label}] {icon} "
                f"{result.get('status','?'):9s} | "
                f"{result.get('request_type','?'):15s} | "
                f"{subject or issue[:50]}"
            )

    print(f"\nDone. Output written to {OUTPUT_CSV}")

    # ── Validate output ───────────────────────────────────────────────────────
    print("\n--- Output validation (first 3 rows) ---")
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as val_f:
        reader = csv.DictReader(val_f)
        assert reader.fieldnames == OUTPUT_FIELDS, (
            f"Column mismatch!\n  expected: {OUTPUT_FIELDS}\n  got: {reader.fieldnames}"
        )
        for j, val_row in enumerate(reader, 1):
            print(f"\n[Row {j}]")
            for field in OUTPUT_FIELDS:
                val = val_row.get(field, "")
                print(f"  {field:<14}: {(val[:80] + '…') if len(val) > 80 else val}")
            if j >= 3:
                break
    print("\nColumns verified:", OUTPUT_FIELDS)


if __name__ == "__main__":
    main()
