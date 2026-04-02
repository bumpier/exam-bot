import unittest

from src.core.auth import credentials_configured, validate_login


class TestAuth(unittest.TestCase):
    def test_credentials_must_be_configured(self) -> None:
        self.assertFalse(credentials_configured("", ""))
        self.assertTrue(credentials_configured("admin", "secret"))

    def test_validate_login_rejects_when_config_missing(self) -> None:
        self.assertFalse(
            validate_login(
                input_username="admin",
                input_password="secret",
                expected_username="",
                expected_password="",
            )
        )

    def test_validate_login_accepts_exact_match(self) -> None:
        self.assertTrue(
            validate_login(
                input_username="admin",
                input_password="secret",
                expected_username="admin",
                expected_password="secret",
            )
        )

    def test_validate_login_rejects_wrong_password(self) -> None:
        self.assertFalse(
            validate_login(
                input_username="admin",
                input_password="wrong",
                expected_username="admin",
                expected_password="secret",
            )
        )


if __name__ == "__main__":
    unittest.main()
