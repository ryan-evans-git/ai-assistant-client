# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's
**Security Advisories** (Security tab → "Report a vulnerability")
rather than opening a public issue. Reports are acknowledged
within 5 business days.

If you can't use GitHub Advisories, email
`ryan-evans-git` via the address listed on their GitHub profile.
PGP-encrypted reports are accepted on request.

## Supported versions

Only the latest released minor version receives security fixes.

## Scope

In scope:

- Code in this repository (`ai_assistant_client/`).
- Provider adapters in `ai_assistant_client/llm/`.
- Container images built from this repository's `Dockerfile`.

Out of scope:

- Vulnerabilities in upstream LLM provider SDKs (`anthropic`,
  `openai`, `google-genai`, `boto3`) — please report those to the
  respective vendors. We track their advisories via Dependabot
  and ship updated pins promptly.
- MCP servers connected via `MCP_SERVERS` — those have their own
  security boundaries.

## What we run on every commit

- Ruff (lint + style)
- Bandit (Python static security analysis, medium+ severity gates merge)
- pip-audit (Python dependency CVE scan)
- Trivy (filesystem + Dockerfile scan)
- CodeQL (GitHub-native SAST, `security-extended` query pack)
- Gitleaks (committed-secret detection across full history)
- Dependabot (weekly dep PRs; immediate security updates)

A failing security check blocks merge to `main`.
