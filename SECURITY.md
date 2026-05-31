# Security Policy

I take the security of forge-infer seriously, even though it is a teaching-grade project. If you find a vulnerability I want to hear about it.

## Reporting a vulnerability

Please email me at **security@sarmalinux.com** with a description of the issue, the steps to reproduce it, and the impact you believe it has. Do not open a public issue for a security problem.

I commit to acknowledging your report within **7 days** and to keeping you updated as I work through a fix. Once a fix is released I am happy to credit you, unless you would rather stay anonymous.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1 | No |

## Scope

forge-infer ships a deterministic stand-in model and is intended for learning and local experimentation, not for serving production traffic. Reports about the HTTP layer, the scheduler, the cache allocator or dependency advisories are all in scope. Reports that depend on pointing the server at untrusted real model weights are out of scope, because that is not a configuration the project supports.
