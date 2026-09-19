# The demo as a served static site.
#
# Build context is ./demo, not the repo root: the image needs the rendered
# payloads and nothing else -- no source, no checkpoints, no bags. That keeps
# the context small and means this image builds on a machine that has only the
# demo folder, which is also how it gets handed to an examiner.
#
#   docker compose up -d demo      ->  http://localhost:8080
#
# The demo also works with no server at all -- open demo/index.html directly.
# Everything is classic scripts and relative paths precisely so that both
# delivery modes work; this image is the http one.

FROM nginx:1.27-alpine

LABEL org.opencontainers.image.title="sim2real-lidar demo" \
      org.opencontainers.image.description="Pre-rendered thesis demo: three LiDAR encoders, simulation and real Ouster data."

COPY nginx-demo.conf /etc/nginx/conf.d/default.conf

# --chown so nginx's unprivileged worker can read them regardless of the
# umask on the machine that built the image.
COPY --chown=nginx:nginx index.html  /usr/share/nginx/html/index.html
COPY --chown=nginx:nginx assets/     /usr/share/nginx/html/assets/
COPY --chown=nginx:nginx payloads/   /usr/share/nginx/html/payloads/

EXPOSE 80

# Fails the container if the shell or the manifest is missing, rather than
# serving a page that renders an empty body and reports nothing.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://localhost/index.html \
   && wget -q -O /dev/null http://localhost/payloads/manifest.js || exit 1
