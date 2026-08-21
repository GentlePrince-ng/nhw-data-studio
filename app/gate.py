"""Access control for the deployed app.

Streamlit Community Cloud has no private apps — that moved to Snowflake — so a
deployed URL is reachable by anyone who has it. This tool takes uploads of health
data, so the URL alone must not be enough.

**It fails closed.** With no code configured the app admits nobody and tells the
owner what to set, rather than serving a wide-open upload form. The alternative
design — open by default with a warning banner — puts a real dataset one
forgotten setting away from a public page, and the banner is exactly what a
hurrying colleague does not read.

Three states:

* `ACCESS_CODE` in the app's secrets → ask for it, once per browser session.
* `NHW_STUDIO_OPEN=1` in the environment → no gate. For running it on your own
  machine, where the gate is friction with nothing on the other side of it.
* Neither → refuse everyone, with the fix printed on the page.

The code is a shared secret for a test group, not an identity: it says *someone
was given this*, not *who*. Rotate it by editing the secret, which logs everyone
out at once. If per-person access matters later, the roster pattern from the NHW
dashboard (hashed codes, one per member) belongs here instead.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

from . import theme

SESSION_KEY = "_nhw_access_ok"


def _configured_codes() -> list:
    """Codes from secrets. `ACCESS_CODE` for one, `ACCESS_CODES` for several."""
    out = []
    try:
        one = st.secrets.get("ACCESS_CODE")
        if one:
            out.append(str(one).strip())
        many = st.secrets.get("ACCESS_CODES")
        if many:
            if isinstance(many, str):
                out += [c.strip() for c in many.split(",")]
            else:
                out += [str(c).strip() for c in many]
    except Exception:
        # No secrets file at all is a normal local state, not an error.
        pass
    return [c for c in out if c]


def _matches(entered: str, codes: list) -> bool:
    # compare_digest on every candidate, and no early return, so a wrong code
    # takes the same time as a right one.
    entered = entered.strip()
    hit = False
    for code in codes:
        if hmac.compare_digest(entered, code):
            hit = True
    return hit


def require() -> bool:
    """Show the gate if needed. True when the page body may render."""
    if os.environ.get("NHW_STUDIO_OPEN") == "1":
        return True
    if st.session_state.get(SESSION_KEY):
        return True

    codes = _configured_codes()
    theme.band("NHW Data Studio", "Nigeria Health Watch")

    if not codes:
        st.error("This app has no access code configured, so it is not letting "
                 "anyone in.")
        st.markdown(
            "**If you own this app:** open it on share.streamlit.io → "
            "*Settings* → *Secrets*, add a line like\n\n"
            "```toml\nACCESS_CODE = \"choose-something-not-guessable\"\n```\n\n"
            "and save. The app restarts on its own. Share that code with the "
            "people who should have access, and change it to revoke.\n\n"
            "**Running it on your own machine?** Set `NHW_STUDIO_OPEN=1` in the "
            "environment and there is no gate at all.")
        return False

    left, mid, right = st.columns([1, 1.5, 1])
    with mid:
        st.markdown("#### Access code")
        st.caption("This tool is being tested by a small group. Ask Solomon for "
                   "the code if you need it.")
        with st.form("gate", clear_on_submit=True):
            entered = st.text_input("Code", type="password",
                                    label_visibility="collapsed",
                                    placeholder="Enter the access code")
            ok = st.form_submit_button("Open", type="primary",
                                       width="stretch")
        # The failure has to be remembered rather than printed inline. A submit
        # button reads True only on the run that handled the submit, and the
        # form clearing itself starts another one — so an inline st.error was
        # drawn and then thrown away, and a mistyped code looked like a dead
        # button. Colleagues will mistype; silence would send them to ask why
        # the app is broken.
        if ok:
            if _matches(entered, codes):
                st.session_state[SESSION_KEY] = True
                st.session_state.pop("_nhw_gate_err", None)
                st.rerun()
            st.session_state["_nhw_gate_err"] = True
        if st.session_state.get("_nhw_gate_err"):
            st.error("That code is not right. Check for a stray space, then "
                     "try again.")
        st.caption("Nothing you upload is stored on the server, and nothing "
                   "leaves this session. Closing the tab ends it.")
    return False
