# Deployment work boundaries

- Keep every addition and generated artifact inside Edge_Deploy. Treat Ahmed's
  source, weights, and data as read-only inputs.
- The user runs experiments themselves. Do not execute benchmarks, inference,
  training or tests unless the user explicitly asks to run them. Static source
  inspection and syntax checks are allowed.
- Keep one comparison script and one Markdown report per method. Use shared
  helpers in _support to avoid duplication.
- Do not generate persistent logs, profiles, JSON duplicates, bytecode caches,
  exported models or engines by default. Use automatically cleaned temporary
  files when export requires them.
- Preserve measured results accurately. Distinguish numerical drift from task
  accuracy and desktop measurements from Jetson validation.
