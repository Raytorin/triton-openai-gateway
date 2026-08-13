# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from scripts.check_python_environment import (
    KNOWN_DECORD_WHEEL_ISSUE,
    has_only_known_decord_issue,
)


def test_known_decord_wheel_issue_is_accepted() -> None:
    assert has_only_known_decord_issue(1, KNOWN_DECORD_WHEEL_ISSUE + "\n")


def test_additional_dependency_conflict_is_rejected() -> None:
    output = "\n".join(
        (
            KNOWN_DECORD_WHEEL_ISSUE,
            "example 1.0 requires dependency>=2, but you have dependency 1.0.",
        )
    )

    assert not has_only_known_decord_issue(1, output)


def test_success_is_not_treated_as_ignored_issue() -> None:
    assert not has_only_known_decord_issue(0, "No broken requirements found.\n")


def test_unexpected_pip_failure_is_rejected() -> None:
    assert not has_only_known_decord_issue(2, KNOWN_DECORD_WHEEL_ISSUE)


def test_similar_message_for_another_version_is_rejected() -> None:
    assert not has_only_known_decord_issue(
        1,
        "decord 0.6.1 is not supported on this platform\n",
    )
