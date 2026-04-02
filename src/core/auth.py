"""
Simple credential validation helpers for Streamlit login gating.
"""

from __future__ import annotations

import hmac


def credentials_configured(expected_username: str, expected_password: str) -> bool:
    return bool(expected_username.strip() and expected_password.strip())


def validate_login(
    input_username: str,
    input_password: str,
    expected_username: str,
    expected_password: str,
) -> bool:
    if not credentials_configured(expected_username, expected_password):
        return False

    return hmac.compare_digest(input_username, expected_username) and hmac.compare_digest(
        input_password, expected_password
    )
