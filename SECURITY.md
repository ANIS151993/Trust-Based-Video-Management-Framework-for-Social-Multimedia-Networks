# Security Policy

## Reporting a vulnerability

If you discover a security issue in the lab pipeline, website, or build tooling in this
repository, please report it privately rather than opening a public issue:

- **Email:** engr.aanis@gmail.com
- **Response time:** best-effort acknowledgment within 5 business days.

Please include: a description of the issue, steps to reproduce, and its potential impact.
Do not include working exploit code in the initial report.

## Scope

This policy covers the code in this repository (`lab/`, `docs/`, build tooling). The
synthetic dataset generator (`lab/data/synthetic_dataset.py`) produces procedural
placeholder imagery only and contains no real weapon photography — see
[`lab/README.md`](lab/README.md) for details.

## Document access policy

The compiled paper (PDF) and the complete LaTeX project are distributed as password-protected
ZIP archives (see the [`#paper`](https://anis151993.github.io/Trust-Based-Video-Management-Framework-for-Social-Multimedia-Networks/#paper)
section of the project site). Both must be requested and unlocked through the site's secure
download gate — no direct, ungated download link to either archive is published anywhere on
the site.

This is a lightweight access-friction measure, **not** a confidentiality guarantee — standard
ZipCrypto encryption is not resistant to a determined offline attacker, and the same `main.tex`
and `main.pdf` remain individually readable in this repository's `paper/` directory (a public
GitHub repository's tracked files cannot be selectively hidden from someone browsing the repo
directly; the gate governs the website's own download flow, not GitHub's file browser). Its
purpose is to let the author track who requests the documents and discourage casual
bulk-scraping through the website, not to guarantee secrecy.

The archive password is shared manually by the author after a short request flow (see the
site's secure download gate) and is never stored in plaintext anywhere in this repository —
only its SHA-256 hash ships in `docs/assets/js/main.js`, used solely for a client-side
confirmation check.
