/* Entry point: read the manifest, build the nav, mount sections lazily. */
(function (DEMO) {
  'use strict';
  var el = DEMO.el;

  function boot() {
    var manifest = DEMO.get('manifest', 'main');
    if (!manifest) {
      document.getElementById('content').appendChild(
        el('p.note', { text: 'manifest not found — the demo payload did not load.' }));
      return;
    }

    /* Every payload URL carries the manifest's build stamp.
     *
     * build_demo.py regenerates payloads in place under the same filenames:
     * frames/0042.png is a different image after a rebuild and metrics.js holds
     * different numbers, so a browser that has any of them cached by name will
     * go on showing the old ones. That is not theoretical -- the page rendered
     * "transcribed, not regenerated at build time" from a cached metrics.js
     * while the server was already serving the measured one.
     *
     * A server can fix this with cache headers, and nginx-demo.conf does. But
     * the demo's two other delivery modes cannot: opened as a file:// folder
     * there is no server at all, and on GitHub Pages the headers are not ours.
     * Putting the stamp in the URL is the only fix that works in all three.
     *
     * index.html and manifest.js themselves still need server-side no-cache --
     * nothing can version the URL that carries the version.
     */
    var build = manifest.generatedAt || manifest.schemaVersion || '1';
    var stamp = '?v=' + encodeURIComponent(build);

    var ctx = {
      manifest: manifest,
      classes: manifest.classes || [],
      metrics: null,
      resolve: function (rel) { return 'payloads/' + rel + stamp; }
    };

    var nav = document.getElementById('nav');
    var main = document.getElementById('content');
    var sections = manifest.sections || [];

    nav.appendChild(el('span.nav-brand', { text: manifest.site.navBrand || 'sim2real lidar' }));
    var navLinks = {};
    sections.filter(function (s) { return s.nav; }).forEach(function (s) {
      var a = el('a', { href: '#' + s.id, text: s.nav });
      navLinks[s.id] = a;
      nav.appendChild(a);
    });

    /* metrics.json backs several sections, so it loads once up front rather
       than per-section. It is small (~25 KB). */
    DEMO.require('metrics', 'main', ctx.resolve('metrics.js'), function (m) {
      ctx.metrics = m;
      mountAll();
    }, function () {
      mountAll();   // sections that need metrics will say so themselves
    });

    /* Each section knows how to mount itself exactly once. Kept in order so a
       jump can force-mount everything ABOVE the target -- see jumpTo(). */
    var entries = [];

    function mountAll() {
      sections.forEach(function (spec, index) {
        var sec = el('section#' + spec.id);
        var head = el('div.section-head');
        if (spec.type !== 'hero') {
          if (spec.nav) head.appendChild(el('span.section-kicker', { text: spec.nav }));
          if (spec.title) head.appendChild(el('h2.section-title', { text: spec.title }));
          if (spec.blurb) head.appendChild(el('div.section-blurb', { html: spec.blurb }));
          sec.appendChild(head);
        }
        var bodyEl = el('div.section-body');
        sec.appendChild(bodyEl);
        main.appendChild(sec);

        var entry = {
          id: spec.id, index: index, node: sec, mounted: false,
          mount: function () {
            if (entry.mounted) return;
            entry.mounted = true;
            DEMO.renderInto(bodyEl, spec, ctx);
          }
        };
        entries.push(entry);

        DEMO.onVisible(sec, { rootMargin: '300px' }, entry.mount);
      });

      buildFooter(manifest);
      setupScrollspy(sections, navLinks);
      wireNavJumps(navLinks);
      honourDeepLink();
    }

    /* Scrolling to a section that hasn't mounted lands in the wrong place:
       the sections above it are still empty, so the target's offset is wrong
       and grows underneath the scroll. Mount everything up to and including
       the target first -- sections below don't affect its position, so they
       stay lazy. */
    function jumpTo(id) {
      var target = null;
      entries.forEach(function (e) { if (e.id === id) target = e; });
      if (!target) return false;
      entries.forEach(function (e) { if (e.index <= target.index) e.mount(); });
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          target.node.scrollIntoView({ behavior: 'auto', block: 'start' });
        });
      });
      return true;
    }

    function wireNavJumps(links) {
      Object.keys(links).forEach(function (id) {
        links[id].addEventListener('click', function (e) {
          e.preventDefault();
          if (jumpTo(id)) history.replaceState(null, '', '#' + id);
        });
      });
      // Jump links inside the hero are created later, so delegate.
      main.addEventListener('click', function (e) {
        var a = e.target.closest && e.target.closest('.hero-jump a');
        if (!a) return;
        var id = a.getAttribute('href').slice(1);
        e.preventDefault();
        if (jumpTo(id)) history.replaceState(null, '', '#' + id);
      });
    }

    function honourDeepLink() {
      var id = null;
      if (location.hash.length > 1) id = location.hash.slice(1);
      var m = /[?&]section=([^&]+)/.exec(location.search);
      if (m) id = decodeURIComponent(m[1]);
      if (id) jumpTo(id);
    }
  }

  function buildFooter(manifest) {
    var site = manifest.site || {};
    var foot = document.getElementById('footer');
    var bits = [];
    if (site.author) bits.push(site.author);
    if (site.date) bits.push(site.date);
    if (manifest.generatedAt) bits.push('built ' + manifest.generatedAt.slice(0, 10));
    foot.appendChild(el('div', { text: bits.join(' · ') }));
    if (site.repoUrl) {
      foot.appendChild(el('div', null, [
        document.createTextNode('Source: '),
        el('a', { href: site.repoUrl, text: site.repoUrl })
      ]));
    }
  }

  function setupScrollspy(sections, navLinks) {
    if (typeof IntersectionObserver !== 'function') return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var a = navLinks[e.target.id];
        if (!a) return;
        if (e.isIntersecting) {
          Object.keys(navLinks).forEach(function (k) { navLinks[k].removeAttribute('aria-current'); });
          a.setAttribute('aria-current', 'true');
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    sections.forEach(function (s) {
      var node = document.getElementById(s.id);
      if (node) io.observe(node);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})(window.DEMO);
