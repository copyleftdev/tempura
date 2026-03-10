# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing **security@tempura-rs.dev** (or open a
[GitHub private security advisory](https://github.com/tempura-rs/tempura/security/advisories/new)
if you prefer).

Include:

- A clear description of the vulnerability
- Steps to reproduce (code, configuration, or inputs if applicable)
- Potential impact assessment
- Any suggested fixes you have in mind

## Response Timeline

| Event | Target |
|---|---|
| Acknowledgement | Within 48 hours |
| Initial assessment | Within 5 business days |
| Fix or mitigation | Within 90 days (critical: 14 days) |
| Public disclosure | After fix is released |

## Scope

Tempura is a pure computation library with **zero runtime dependencies** and
`#![forbid(unsafe_code)]`. The attack surface is correspondingly narrow:

**In scope:**
- Correctness bugs that produce silently wrong numerical results
- Determinism violations (same seed, different output)
- Denial-of-service via crafted inputs (panic, infinite loop, stack overflow)
- Dependency vulnerabilities (dev-dependencies included)

**Out of scope:**
- Issues in downstream code using tempura
- Performance regressions without correctness impact
- Theoretical statistical weaknesses in the PRNGs that do not affect practical use

## Disclosure Policy

We follow a **coordinated disclosure** model. We will credit reporters in the
release notes and CHANGELOG unless you prefer to remain anonymous.
