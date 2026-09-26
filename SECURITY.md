# Security Policy

**Language:** English | [Русский](SECURITY.ru.md)

## Reporting A Vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private security advisory feature for the repository and include:

- the affected endpoint or component;
- the tested revision and deployment mode;
- reproduction steps or a minimal request;
- expected impact;
- relevant logs with credentials and user data removed.

If private advisories are not enabled, contact the repository owner through a
private channel listed on their GitHub profile before disclosing details.

## Supported Versions

Security fixes are applied to the current default branch. Historical release
copies are not maintained unless explicitly stated in a GitHub release.

## Deployment Boundary

This project does not provide authentication, tenant isolation, TLS termination,
or a secret manager. Deploy it behind an authenticated proxy and keep raw
Triton, metrics, DCGM, and tracing endpoints on trusted networks.

Model repositories are part of the trusted computing base. With
`trust_remote_code=true`, tokenizer or model files may execute Python code in the
Triton container. Only load reviewed models from controlled repositories.

Remote media, large request bodies, PDF parsing, video decoding, and long-lived
streams are resource-sensitive inputs. Keep the default network restrictions and
configure hard size, duration, pixel, queue, and timeout limits for your
environment.

## Temporary Dependency Audit Exception

For issue #28, CI temporarily accepts **PYSEC-2026-3804** (aliases
GHSA-4j2p-28q2-5m79 / CVE-2026-69112) only while the runtime requirements pin
`accelerate==1.14.0`. The exception expires at **2026-10-10 00:00 UTC**;
`scripts/audit_dependencies.py` then automatically runs the unmodified audit.
Changing the package version also removes this exception. Other advisories,
other dependencies and scanner failures remain blocking. Test requirements
receive no exception. CI prints the exception whenever it is used.

This is temporary risk acceptance, not a fix. Malicious checkpoint shard indexes
can reference files outside the model directory or special files that block the
loader. Only load reviewed models from controlled repositories. Keep this gate
separate from release approval; replace the exception with a verified upstream
fix or a tested patch. Accelerate 1.15.0 clears the current advisory version range
but retains the affected loader code, so a version bump alone is insufficient.

See the [advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79) and
[upstream fix proposal](https://github.com/huggingface/accelerate/pull/4214).
