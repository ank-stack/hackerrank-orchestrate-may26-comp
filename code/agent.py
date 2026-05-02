import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """\
You are a support triage agent for HackerRank, Claude (Anthropic), and Visa.

Output JSON with exactly these fields:
- status: "replied" | "escalated"
- product_area: e.g. screen, community, privacy, general_support, travel_support, billing, conversation_management, account_management, team-and-enterprise, integrations, interviews, identity-management, safeguards, pro-and-max-plans, claude-api, amazon-bedrock, claude-code, skillup, engage, library, settings
- response: user-facing answer grounded ONLY in corpus snippets; never invent facts, policies, or phone numbers
- justification: one sentence explaining the decision
- request_type: "product_issue" | "feature_request" | "bug" | "invalid"

Escalate when: identity theft, fraud, billing dispute needing human action, account access removed by admin, third-party action required (recruiter/merchant), insufficient info to diagnose, multi-user security incident.
Reply (do not escalate) for: standard FAQs answerable from corpus, security vuln reports (acknowledge + bug bounty), out-of-scope/harmful requests (reply as invalid).
request_type: product_issue=can't do what product supports; feature_request=doesn't exist yet; bug=technical error; invalid=off-topic/harmful/unreasonable.

Example:
TICKET: Issue="What is the name of the actor in Iron Man?" | Company="None"
OUTPUT: {"status":"replied","product_area":"conversation_management","response":"This is outside the scope of my support capabilities.","justification":"Completely off-topic.","request_type":"invalid"}

Use ONLY provided corpus snippets. If insufficient, escalate. Output JSON only — no markdown, no extra text.
"""

_TOOL = {
    "name": "process_ticket",
    "description": "Return structured triage decision for a support ticket",
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["replied", "escalated"]},
            "product_area": {"type": "string"},
            "response": {"type": "string"},
            "justification": {"type": "string"},
            "request_type": {
                "type": "string",
                "enum": ["product_issue", "feature_request", "bug", "invalid"],
            },
        },
        "required": ["status", "product_area", "response", "justification", "request_type"],
    },
}


def _build_user_message(issue: str, subject: str, company: str, docs: list[dict]) -> str:
    parts = [f"TICKET:\nIssue: {issue}\nSubject: {subject}\nCompany: {company}\n"]
    if docs:
        parts.append("CORPUS SNIPPETS (use these as your only source):")
        for i, doc in enumerate(docs, 1):
            rel_path = doc["path"].replace("\\", "/").split("/data/")[-1]
            parts.append(f"\n[{i}] Source: data/{rel_path}\n{doc['snippet']}")
    else:
        parts.append("No corpus snippets found — escalate if uncertain.")
    return "\n".join(parts)


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    docs: list[dict],
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    user_content = _build_user_message(issue, subject, company, docs)

    response = _client.messages.create(
        model=model,
        max_tokens=450,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "process_ticket"},
        messages=[{"role": "user", "content": user_content}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "process_ticket":
            return block.input

    # Fallback: escalate if structured output missing
    return {
        "status": "escalated",
        "product_area": "general_support",
        "response": "Escalate to a human",
        "justification": "Agent failed to produce a structured response.",
        "request_type": "bug",
    }
