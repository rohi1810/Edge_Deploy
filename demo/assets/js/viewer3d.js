/* Per-scan 3D viewer: the real scan coloured by semantic class, with the
 * latent-only reconstruction overlaid.
 *
 * This is the claim the reconstruction metric is making, shown directly: the
 * red points are decoded from the 256-d latent alone, with no skip connections
 * back to the input, so how well they trace the corridor and the objects is
 * what the bottleneck actually preserved.
 *
 * Three separate THREE.Points objects rather than one, so environment,
 * objects and reconstruction each get their own size and opacity without a
 * custom shader.
 */
(function (DEMO) {
  'use strict';
  var el = DEMO.el;

  function classColor(ctx, id) {
    var hit = null;
    (ctx.classes || []).forEach(function (c) { if (c.id === id) hit = c; });
    return hit ? hit.color : '#888888';
  }

  function classNameOf(ctx, id) {
    var hit = null;
    (ctx.classes || []).forEach(function (c) { if (c.id === id) hit = c; });
    return hit ? hit.name : ('Class ' + id);
  }

  function createViewer(host, payload, ctx) {
    if (typeof THREE === 'undefined') {
      host.appendChild(el('p.note', { text: '3D library unavailable.' }));
      return null;
    }

    var canvasWrap = el('div.viewer-canvas');
    var viewport = el('div.viewer.card', null, [canvasWrap]);

    var sampleBtns = [];
    var sampleList = el('div');
    var toggles = {};
    function toggleRow(key, label, checked) {
      var input = el('input', { type: 'checkbox', checked: checked ? true : null });
      toggles[key] = input;
      return el('label.viewer-toggle', null, [input, el('span', { text: label })]);
    }

    var sizeSlider = el('input.viewer-slider', {
      type: 'range', min: '0.01', max: '0.15', step: '0.005', value: '0.045',
      'aria-label': 'Point size'
    });

    var caption = el('p.caption');

    var panel = el('div.viewer-panel', null, [
      el('div', null, [el('h3', { text: 'Sample' }), sampleList]),
      el('div', null, [
        el('h3', { text: 'Layers' }),
        toggleRow('env', 'Environment', true),
        toggleRow('objects', 'Objects', true),
        toggleRow('recon', 'Reconstruction (latent only)', true)
      ]),
      el('div', null, [el('h3', { text: 'Point size' }), sizeSlider]),
      el('div', null, [
        el('h3', { text: 'View' }),
        el('div', { style: 'display:flex;gap:6px;flex-wrap:wrap' }, [
          el('button.viewer-btn', { type: 'button', text: 'Top', 'data-view': 'top' }),
          el('button.viewer-btn', { type: 'button', text: 'Side', 'data-view': 'side' }),
          el('button.viewer-btn', { type: 'button', text: 'Reset', 'data-view': 'reset' })
        ])
      ])
    ]);

    var root = el('div.viewer-layout', null, [viewport, panel]);
    host.appendChild(root);
    host.appendChild(caption);

    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (err) {
      DEMO.clear(host);
      host.appendChild(el('p.note', {
        text: 'WebGL is unavailable in this browser, so the interactive point cloud ' +
              'cannot render. The equivalent static figures are in the scan explorer above.' }));
      return null;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    canvasWrap.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.05, 400);
    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    var group = new THREE.Group();
    scene.add(group);
    var current = null;

    function buildSample(sample) {
      while (group.children.length) {
        var c = group.children.pop();
        c.geometry.dispose();
        c.material.dispose();
      }

      var pos = new Float32Array(sample.input.positions);
      var ids = sample.input.classIds;
      var envIdx = [], objIdx = [];
      for (var i = 0; i < ids.length; i++) (ids[i] === 0 ? envIdx : objIdx).push(i);

      function subset(indices, withColor) {
        var p = new Float32Array(indices.length * 3);
        var col = withColor ? new Float32Array(indices.length * 3) : null;
        var tmp = new THREE.Color();
        for (var k = 0; k < indices.length; k++) {
          var s = indices[k] * 3;
          p[k * 3] = pos[s]; p[k * 3 + 1] = pos[s + 1]; p[k * 3 + 2] = pos[s + 2];
          if (col) {
            tmp.set(classColor(ctx, ids[indices[k]]));
            col[k * 3] = tmp.r; col[k * 3 + 1] = tmp.g; col[k * 3 + 2] = tmp.b;
          }
        }
        var g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(p, 3));
        if (col) g.setAttribute('color', new THREE.BufferAttribute(col, 3));
        return g;
      }

      var envPts = new THREE.Points(subset(envIdx, false), new THREE.PointsMaterial({
        size: 0.03, sizeAttenuation: true, transparent: true, opacity: 0.3,
        color: new THREE.Color(getComputedStyle(document.documentElement)
          .getPropertyValue('--text-dim').trim() || '#7c8894')
      }));
      envPts.userData.role = 'env';

      var objPts = new THREE.Points(subset(objIdx, true), new THREE.PointsMaterial({
        size: 0.09, sizeAttenuation: true, vertexColors: true
      }));
      objPts.userData.role = 'objects';

      var rg = new THREE.BufferGeometry();
      rg.setAttribute('position', new THREE.BufferAttribute(
        new Float32Array(sample.recon.positions), 3));
      var reconPts = new THREE.Points(rg, new THREE.PointsMaterial({
        size: 0.07, sizeAttenuation: true, transparent: true, opacity: 0.85,
        depthWrite: false,
        color: new THREE.Color(getComputedStyle(document.documentElement)
          .getPropertyValue('--danger').trim() || '#e8705a')
      }));
      reconPts.userData.role = 'recon';

      group.add(envPts, objPts, reconPts);
      current = sample;
      applyToggles();
      applySize();
      resetView();

      var counts = Object.keys(sample.classCounts).map(function (cid) {
        return classNameOf(ctx, +cid) + ' ' + sample.classCounts[cid].toLocaleString();
      }).join(' · ');
      caption.textContent = sample.sampleId + ' — ' + counts +
        ' · red = ' + sample.recon.count.toLocaleString() +
        ' points decoded from the 256-d latent alone';
      requestRender();
    }

    function applyToggles() {
      group.children.forEach(function (c) {
        var t = toggles[c.userData.role];
        c.visible = t ? t.checked : true;
      });
      requestRender();
    }
    function applySize() {
      var s = parseFloat(sizeSlider.value);
      group.children.forEach(function (c) {
        var role = c.userData.role;
        c.material.size = role === 'env' ? s * 0.6 : role === 'recon' ? s * 1.5 : s * 2;
      });
      requestRender();
    }

    function span() {
      var b = current.bounds;
      return Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    }
    function setView(mode) {
      var c = current.center, d = span() * 0.75;
      controls.target.set(c[0], c[1], c[2]);
      if (mode === 'top') camera.position.set(c[0], c[1], c[2] + d);
      else if (mode === 'side') camera.position.set(c[0], c[1] - d, c[2] + d * 0.12);
      else camera.position.set(c[0] + d * 0.7, c[1] - d * 0.7, c[2] + d * 0.45);
      camera.far = span() * 10;
      camera.updateProjectionMatrix();
      controls.update();
      requestRender();
    }
    function resetView() { setView('reset'); }

    panel.querySelectorAll('[data-view]').forEach(function (b) {
      b.addEventListener('click', function () { setView(b.getAttribute('data-view')); });
    });
    Object.keys(toggles).forEach(function (k) {
      toggles[k].addEventListener('change', applyToggles);
    });
    sizeSlider.addEventListener('input', applySize);

    payload.samples.forEach(function (s, i) {
      var b = el('button.viewer-sample', { type: 'button', 'aria-current': i === 0 ? 'true' : 'false' }, [
        el('span.swatch', { style: 'background:' + classColor(ctx, s.highlightClass) }),
        el('span', { text: classNameOf(ctx, s.highlightClass) }),
        el('span.mono', { style: 'margin-left:auto;font-size:0.72rem',
                          text: (s.classCounts[s.highlightClass] || 0).toLocaleString() })
      ]);
      b.addEventListener('click', function () {
        sampleBtns.forEach(function (x, j) { x.setAttribute('aria-current', j === i ? 'true' : 'false'); });
        buildSample(s);
      });
      sampleBtns.push(b);
      sampleList.appendChild(b);
    });

    var running = false, visible = false;
    function loop() { if (!running) return; controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); }
    function requestRender() { if (!running) { controls.update(); renderer.render(scene, camera); } }
    function start() { if (!running) { running = true; loop(); } }
    function stop() { running = false; }

    function resize() {
      var w = canvasWrap.clientWidth || 900;
      var h = Math.max(320, Math.round(w * 9 / 16));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender();
    }
    if (typeof ResizeObserver === 'function') new ResizeObserver(resize).observe(canvasWrap);
    resize();

    DEMO.observeVisibility(root, function (vis) { visible = vis; vis ? start() : stop(); });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stop(); else if (visible) start();
    });

    buildSample(payload.samples[0]);
    return { el: root, dispose: stop };
  }

  DEMO.createViewer = createViewer;
})(window.DEMO);
