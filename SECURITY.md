# Security Policy

## Supported versions

Only the latest release on the main branch receives security fixes.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately:

- **Email:** hello@thrindex.com
- **Subject line:** `[SECURITY] <one-line description>`

You will receive an acknowledgement within **72 hours** and a status update
within **14 days**. Critical vulnerabilities affecting shipped artifacts are
eligible for a coordinated disclosure window of up to **90 days** before
public disclosure.

We commit to a public post-mortem for any finding that affected a shipped artifact.

## Scope

| In scope | Out of scope |
|---|---|
| `thrindex-numerics` and all open crates | Customer-proprietary configurations |
| The `.thx` artifact format and signing chain | Third-party vendor SDKs |
| The public conformance suite | Issues in end-user models |

## Security commitments

- `cargo audit` and `cargo deny` gate every CI merge.
- Reproducible builds: every tagged release can be rebuilt bit-for-bit.
- SBOM published with every release.
- The SDK phones home nothing by default; telemetry is opt-in with a documented, public schema.
