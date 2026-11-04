# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Immutable references for workflow jobs and steps, parsed as YAML."""

from pathlib import PurePosixPath
import re


COMMIT_REFERENCE = re.compile(r"[^\s@]+/[^\s@]+@[0-9a-fA-F]{40}\Z")
CONTAINER_REFERENCE = re.compile(r"docker://[^\s@]+@sha256:[0-9a-fA-F]{64}\Z")


def _immutable_reference(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("./"):
        return value != "./" and ".." not in PurePosixPath(value).parts
    if value.startswith("docker://"):
        return CONTAINER_REFERENCE.fullmatch(value) is not None
    return COMMIT_REFERENCE.fullmatch(value) is not None


def action_reference_errors(content: str, source: str = "workflow") -> list[str]:
    # ruamel.yaml is pinned with the CI test dependencies.
    # Keep this import out of the dependency-free early source checker.
    from ruamel.yaml import YAML, YAMLError

    try:
        workflow = YAML(typ="safe", pure=True).load(content)
    except YAMLError:
        return [f"{source}: unable to parse workflow YAML"]
    if not isinstance(workflow, dict) or not isinstance(workflow.get("jobs"), dict):
        return [f"{source}: workflow jobs must be a mapping"]
    errors = []
    for name, job in workflow["jobs"].items():
        if not isinstance(job, dict):
            continue  # Workflow schema is checked by actionlint.
        references = []
        if "uses" in job:
            references.append((f"job {name!r}", job["uses"]))
        steps = job.get("steps", [])
        if isinstance(steps, list):
            references.extend(
                (f"job {name!r}, step {index}", step["uses"])
                for index, step in enumerate(steps, 1)
                if isinstance(step, dict) and "uses" in step
            )
        for location, reference in references:
            if not _immutable_reference(reference):
                errors.append(f"{source}: {location} requires an immutable action reference: {reference!r}")
    return errors
