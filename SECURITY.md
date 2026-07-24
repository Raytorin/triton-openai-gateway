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
