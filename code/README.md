# Support Ticket Triage Agent

A hybrid pipeline that reads `support_tickets/support_tickets.csv` and writes classified, corpus-grounded responses to `support_tickets/output.csv`. The design prioritises accuracy and minimal API usage through a layered decision system.

## Architecture

```
main.py        — rule engine → cache → retrieval gate → hybrid LLM → CSV writer + validator
retriever.py   — TF-IDF retrieval (top_k=2, ~380-char snippets) over 774-doc corpus in data/
agent.py       — Claude Haiku or Sonnet with cached system prompt + forced tool-use output
```

## Pipeline per ticket

| Step | What happens | API used? |
|------|-------------|-----------|
| 1. Company resolution | Normalise `Company`; infer from content if `None` | No |
| 2. Rule engine | Match against 11 keyword groups (outage, identity theft, invalid, bug bounty, score dispute, billing, payment fail, login, password, subscription, how-to) | No |
| 3. Cache lookup | Identical `(issue, subject)` pairs reuse the prior result | No |
| 4. TF-IDF retrieval | Top-2 corpus snippets (~380 chars), filtered by company | No |
| 5. Score gate | Best score < 0.05 → escalate without calling API | No |
| 6. Hybrid LLM | issue > 200 chars → Sonnet; otherwise → Haiku | **Yes** |
| 7. Write + validate | Append row; assert columns; print first 3 rows at end | No |

## Rule coverage (Task 1 & 4)

| Rule group | Keywords (sample) | Output |
|------------|-------------------|--------|
| Outage | "is down", "all requests failing" | escalated / bug |
| Identity theft | "identity theft", "identity stolen" | escalated / product_issue |
| Harmful / invalid | "delete all files", "règles internes" | replied / invalid |
| Bug bounty | "bug bounty", "security vulnerability" | replied / bug |
| Score dispute | "increase my score", "review my answers" | replied / invalid |
| Urgent billing | "refund asap", "refund today" | escalated / product_issue |
| Payment failed | "payment failed", "card declined" | replied / product_issue |
| Login issues | "can't login", "account locked" | replied / product_issue |
| Password reset | "forgot password", "reset password" | replied / product_issue |
| Subscription | "pause subscription", "cancel plan" | replied / product_issue |
| Generic how-to | "how do i", "where can i" (no financial nouns) | replied / product_issue |

## Hybrid model logic

```python
model = "claude-sonnet-4-6"       if len(issue) > 200 else
        "claude-haiku-4-5-20251001"
```

Long or complex tickets get Sonnet's stronger reasoning; short/simple tickets use Haiku at ~20× lower cost. The choice is logged per row as `[SONNET]` or `[HAIKU ]`.

## Test mode

Set `TEST_MODE = True` at the top of `main.py` to process 10 randomly sampled rows instead of the full dataset. Useful for fast iteration during development.

```python
# code/main.py — line 19
TEST_MODE = True   # ← flip to False before final submission run
```

## Escalation logic

Escalate only for: service outage, identity theft, fraud investigation, urgent billing dispute requiring human action, admin-removed account access, or tickets with no corpus support (score < 0.05).

Normal user questions, FAQs, and known platform limitations are replied to directly — not escalated unnecessarily.

## Setup

```bash
# 1. Install dependencies
pip install anthropic scikit-learn python-dotenv

# 2. Configure API key
cp .env.example .env      # from repo root
# Edit .env → ANTHROPIC_API_KEY=sk-ant-...

# 3. Full run
python code/main.py

# 4. Quick test (10 samples)
# Set TEST_MODE = True in code/main.py, then:
python code/main.py
```

## Output schema

| Column        | Values |
|---------------|--------|
| issue         | original ticket body |
| subject       | original subject line |
| company       | original company field |
| response      | user-facing answer grounded in corpus |
| product_area  | most relevant support category |
| status        | `replied` or `escalated` |
| request_type  | `product_issue`, `feature_request`, `bug`, `invalid` |
| justification | concise explanation of the routing decision |

## Dependencies

- `anthropic` — Claude API SDK (Haiku + Sonnet)
- `scikit-learn` — TF-IDF vectorizer and cosine similarity
- `python-dotenv` — load `.env` file
