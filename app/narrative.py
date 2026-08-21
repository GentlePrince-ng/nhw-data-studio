"""Optional: ask Claude to turn the computed findings into a written brief.

Strictly a layer on top of `insights.py`, never a replacement for it. Every
number in the prompt has already been computed from the file, so the model is
writing prose about arithmetic it did not do — which is the only way to use an
LLM here without putting invented figures into an NHW document. The app works
fully with this switched off.

**What leaves the machine.** The aggregates only: the column map, the finding
sentences, and the small summary tables already visible on screen. No row of
respondent data is ever sent, and the panel says so above the button. That
matters because the uploads this tool is built for include named health workers
and household survey responses.

Requires `pip install anthropic` and a key (ANTHROPIC_API_KEY, or a
`.streamlit/secrets.toml` entry, or an `ant auth login` profile).
"""
from __future__ import annotations

import hashlib
import os

MODEL = "claude-opus-5"
MAX_TOKENS = 8000
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """You write short analytical briefs for Nigeria Health Watch, a health \
communications and advocacy organisation in Abuja.

You are given a profile of a dataset and a list of findings that have ALREADY \
been computed from it. Your job is to write the brief, not to analyse the data.

Rules, in order of importance:
1. Never state a number that is not in the material you were given. Do not \
estimate, extrapolate, annualise, or compute anything new — not even a total.
2. If the findings flag a data-quality problem, lead with it. A reader who \
quotes a figure built on duplicated rows or on a column that is 60% empty has \
been failed by the brief.
3. Say what the data cannot tell you. Coverage gaps, missing states, absent \
denominators and self-selected samples all limit what may be claimed.
4. Never imply causation from an association.
5. Write for a programme lead, not a statistician: plain sentences, no jargon, \
no bullet-point padding, British spelling, Nigerian context assumed.

Structure your reply as markdown with exactly these sections:
**What this data is** — two or three sentences.
**Read these first** — the data-quality caveats that change interpretation.
**What it shows** — the substantive patterns.
**What to ask next** — three or four specific questions or checks, each one \
answerable with this dataset or with a named additional source."""


def available() -> tuple[bool, str]:
    """(usable, why not). Checked before the panel offers to run anything."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "The `anthropic` package is not installed (pip install anthropic)."
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False, ("No API key found. Set ANTHROPIC_API_KEY in the "
                       "environment or in .streamlit/secrets.toml.")
    return True, ""


def build_payload(prof, findings, extra_tables=None) -> str:
    """The prompt body: aggregates only, in a shape that is cheap to read."""
    lines = [f"# Dataset: {prof.source}" + (f" (sheet {prof.sheet})" if prof.sheet else ""),
             f"Records: {prof.rows:,}. Columns: {len(prof.columns)}."]
    if prof.truncated:
        lines.append(f"NOTE: {prof.truncated:,} further rows in the file were "
                     "not read, so every figure below covers only the rows read.")

    lines.append("\n## Columns, as typed by the tool")
    for c in prof.columns.values():
        bit = (f"- {c.name} | reads as {c.role} | {c.filled:,} filled "
               f"({c.missing_pct:.0f}% blank) | {c.unique:,} distinct")
        if c.role == "numeric" and c.stats:
            bit += (f" | mean {c.stats.get('mean', 0):,.2f}, median "
                    f"{c.stats.get('median', 0):,.2f}, range "
                    f"{c.stats.get('min', 0):,.2f}–{c.stats.get('max', 0):,.2f}")
        if c.sentinel_hits:
            bit += (" | sentinel codes present: "
                    + ", ".join(f"{k} x{v}" for k, v in c.sentinel_hits.items()))
        if c.role in ("categorical", "geo", "binary") and c.stats.get("top"):
            bit += f" | most common: {c.stats['top']} ({c.stats.get('top_n', 0):,})"
        lines.append(bit)

    lines.append("\n## Computed findings, most consequential first")
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. [{f.kind} / {f.severity}] {f.text}")

    for title, table in (extra_tables or []):
        if table is None or not len(table):
            continue
        lines.append(f"\n## Table: {title}")
        lines.append(table.head(25).to_csv(index=False).strip())

    lines.append("\nWrite the brief now, following the four sections exactly.")
    return "\n".join(lines)


def digest(payload: str) -> str:
    """Cache key — an identical payload must never be billed twice."""
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def write_brief(payload: str, effort: str = "medium") -> tuple[str, str]:
    """Return (markdown, note). Any failure comes back as a message, not a crash.

    Refusal fallbacks are on: if the safety classifier declines the request, the
    API re-runs it on a fallback model inside the same call rather than handing
    back nothing. An older SDK that rejects the beta is retried without it.
    """
    import anthropic

    client = anthropic.Anthropic()
    args = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        messages=[{"role": "user", "content": payload}],
    )
    note = ""
    try:
        resp = client.beta.messages.create(
            betas=[FALLBACK_BETA], fallbacks="default", **args)
    except (anthropic.BadRequestError, TypeError):
        # Older SDK or beta unavailable — the brief matters more than the rescue.
        resp = client.messages.create(**args)
        note = "Refusal fallback unavailable in this SDK version."
    except anthropic.RateLimitError:
        return "", "Rate limited by the API. Wait a moment and try again."
    except anthropic.APIStatusError as exc:
        return "", f"API error {exc.status_code}: {exc.message}"
    except anthropic.APIConnectionError:
        return "", "Could not reach the API — check the network connection."

    if resp.stop_reason == "refusal":
        return "", "The request was declined by the API safety system."
    text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    if resp.stop_reason == "max_tokens":
        note = (note + " Output hit the token cap and may be cut short.").strip()
    used = getattr(resp, "usage", None)
    if used:
        note = (f"{note} {used.input_tokens:,} in / {used.output_tokens:,} out "
                f"tokens on {resp.model}.").strip()
    return text, note
