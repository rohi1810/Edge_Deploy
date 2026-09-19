/* Tunnel reconstruction: the SLAM point cloud of the corridor, then a cut to
 * the reconstructed mesh.
 *
 * A cut, not a morph. The two assets are not in a shared coordinate frame --
 * the cloud spans 316x100x18.7 m and the mesh 128x68x4.2 m, and the mesh's Y
 * extends past the cloud's -- so cross-fading them in place would fling points
 * to somewhere the surface isn't. Each is framed from its own bounding box and
 * the sequence moves between them deliberately.
 *
 * Both are pre-computed: the mesh is a real CloudCompare reconstruction, shown
 * here rather than recomputed. The stage labels say so; there is deliberately
 * no progress bar implying live work.
 */
(function (DEMO) {
  'use strict';
  var el = DEMO.el;

  var STAGES = [
    { key: 'cloud', label: 'Raw SLAM point cloud',
      note: 'Registered LiDAR returns from the drive, coloured by return intensity.' },
    { key: 'dense', label: 'Accumulated and filtered',
      note: 'The full registered cloud, viewed along the corridor.' },
    { key: 'surface', label: 'Surface reconstruction',
      note: 'Poisson surface reconstruction (CloudCompare), pre-computed.' },
    { key: 'mesh', label: 'Reconstructed mesh',
      note: 'The watertight corridor mesh the simulation environment was built from.' }
  ];

  function decodePositions(block) {
    // int16 + per-axis scale/offset -> Float32Array. Halves payload size at
    // ~5mm error over a 300m span, which is invisible at these point sizes.
    var q = new Int16Array(DEMO.b64ToBytes(block.positions).buffer);
    var n = block.count !== undefined ? block.count : block.vertexCount;
    var out = new Float32Array(n * 3);
    var s = block.scale, o = block.offset;
    for (var i = 0; i < n; i++) {
      out[i * 3] = (q[i * 3] + 32767) * s[0] + o[0];
      out[i * 3 + 1] = (q[i * 3 + 1] + 32767) * s[1] + o[1];
      out[i * 3 + 2] = (q[i * 3 + 2] + 32767) * s[2] + o[2];
    }
    return out;
  }

  /* viridis-ish ramp: dark blue -> teal -> yellow. Matches the range images
     elsewhere on the page so intensity reads as "sensor data" consistently. */
  function intensityColors(bytes) {
    var stops = [[0.267,0.005,0.329],[0.188,0.408,0.557],[0.208,0.718,0.473],[0.993,0.906,0.144]];
    var c = new Float32Array(bytes.length * 3);
    for (var i = 0; i < bytes.length; i++) {
      var t = bytes[i] / 255 * (stops.length - 1);
      var lo = Math.min(Math.floor(t), stops.length - 2), f = t - lo;
      var a = stops[lo], b = stops[lo + 1];
      c[i * 3] = a[0] + (b[0] - a[0]) * f;
      c[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
      c[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
    }
    return c;
  }

  function createTunnel(host, payload, ctx) {
    if (typeof THREE === 'undefined') {
      host.appendChild(el('p.note', { text: '3D library unavailable.' }));
      return null;
    }

    var stageLabel = el('span.stage-label', { text: STAGES[0].label });
    var stageNote = el('span.stage-note', { text: STAGES[0].note });
    var dots = STAGES.map(function (s, i) {
      return el('button.stage-dot', { type: 'button', 'aria-label': s.label,
                                       'data-i': i, 'aria-current': i === 0 ? 'true' : 'false' });
    });
    var replay = el('button.viewer-btn', { type: 'button', text: 'Replay' });

    var canvasWrap = el('div.viewer-canvas');
    var bar = el('div.viewer-bar', null, [
      el('div.stage-info', null, [stageLabel, stageNote]),
      el('div.stage-dots', null, dots),
      replay
    ]);
    var root = el('div.viewer.card', null, [canvasWrap, bar]);
    host.appendChild(root);

    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (err) {
      // Software rendering, remote desktop, or a locked-down machine. Say so
      // rather than showing an empty black box.
      DEMO.clear(root);
      root.appendChild(el('p.note', {
        text: 'WebGL is unavailable in this browser, so the interactive reconstruction ' +
              'cannot render. The reconstruction itself is unaffected — it is a ' +
              'pre-computed CloudCompare result.' }));
      return null;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    canvasWrap.appendChild(renderer.domElement);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(55, 16 / 9, 0.1, 2000);
    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    var key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(1, 2, 1.5);
    scene.add(key);

    // ---- point cloud -----------------------------------------------------
    var cg = new THREE.BufferGeometry();
    cg.setAttribute('position', new THREE.BufferAttribute(decodePositions(payload.cloud), 3));
    if (payload.cloud.intensity) {
      cg.setAttribute('color', new THREE.BufferAttribute(
        intensityColors(DEMO.b64ToBytes(payload.cloud.intensity)), 3));
    }
    var cloud = new THREE.Points(cg, new THREE.PointsMaterial({
      size: 0.12, sizeAttenuation: true, transparent: true, opacity: 1,
      vertexColors: !!payload.cloud.intensity,
      color: payload.cloud.intensity ? 0xffffff : 0x4fd1c5
    }));
    scene.add(cloud);

    // ---- mesh ------------------------------------------------------------
    var mg = new THREE.BufferGeometry();
    mg.setAttribute('position', new THREE.BufferAttribute(decodePositions(payload.mesh), 3));
    mg.setIndex(new THREE.BufferAttribute(
      new Uint32Array(DEMO.b64ToBytes(payload.mesh.indices).buffer), 1));
    if (payload.mesh.colors) {
      var cb = DEMO.b64ToBytes(payload.mesh.colors);
      var mc = new Float32Array(cb.length);
      for (var i = 0; i < cb.length; i++) mc[i] = cb[i] / 255;
      mg.setAttribute('color', new THREE.BufferAttribute(mc, 3));
    }
    mg.computeVertexNormals();
    var mesh = new THREE.Mesh(mg, new THREE.MeshStandardMaterial({
      vertexColors: !!payload.mesh.colors,
      color: payload.mesh.colors ? 0xffffff : 0x8899a6,
      roughness: 0.95, metalness: 0.0, side: THREE.DoubleSide,
      transparent: true, opacity: 0
    }));
    mesh.visible = false;
    scene.add(mesh);

    function frame(block, azimuth) {
      var c = block.center, b = block.bounds;
      var span = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
      var d = span * 0.62;
      controls.target.set(c[0], c[1], c[2]);
      camera.position.set(c[0] + Math.cos(azimuth) * d, c[1] + Math.sin(azimuth) * d,
                          c[2] + span * 0.28);
      camera.near = Math.max(0.05, span / 5000);
      camera.far = span * 12;
      camera.updateProjectionMatrix();
      controls.update();
    }
    frame(payload.cloud, 0.6);

    // ---- stages ----------------------------------------------------------
    var stage = 0;
    function setStage(i, animate) {
      i = Math.max(0, Math.min(STAGES.length - 1, i));
      stage = i;
      stageLabel.textContent = STAGES[i].label;
      stageNote.textContent = STAGES[i].note;
      dots.forEach(function (d, k) { d.setAttribute('aria-current', k === i ? 'true' : 'false'); });

      var showMesh = i >= 2;
      mesh.visible = showMesh;
      mesh.material.opacity = i >= 3 ? 1 : (i === 2 ? 0.55 : 0);
      cloud.material.opacity = i >= 3 ? 0.0 : (i === 2 ? 0.35 : 1);
      cloud.visible = cloud.material.opacity > 0.01;
      cloud.material.size = i === 0 ? 0.12 : 0.09;

      if (animate !== false) frame(i >= 2 ? payload.mesh : payload.cloud, i === 1 ? 2.2 : 0.6);
      requestRender();
    }
    dots.forEach(function (d) {
      d.addEventListener('click', function () { setStage(+d.getAttribute('data-i')); });
    });

    /* Scroll through the section drives the stage. A replay button exists
       regardless: at a defense you need to re-show this without scrolling
       away and back. */
    function onScroll() {
      var r = root.getBoundingClientRect();
      var total = r.height + window.innerHeight;
      var p = (window.innerHeight - r.top) / total;      // 0 entering, 1 leaving
      var idx = Math.floor(Math.max(0, Math.min(0.999, (p - 0.12) / 0.62)) * STAGES.length);
      if (idx !== stage) setStage(idx);
    }

    var running = false, visible = false;
    function loop() {
      if (!running) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    }
    function requestRender() {
      if (!running) { controls.update(); renderer.render(scene, camera); }
    }
    function start() { if (!running) { running = true; loop(); } }
    function stop() { running = false; }

    function resize() {
      var w = canvasWrap.clientWidth || 960;
      var h = Math.max(320, Math.round(w * 9 / 16));
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender();
    }
    if (typeof ResizeObserver === 'function') new ResizeObserver(resize).observe(canvasWrap);
    resize();

    DEMO.observeVisibility(root, function (vis) {
      visible = vis;
      if (vis) { start(); window.addEventListener('scroll', onScroll, { passive: true }); }
      else { stop(); window.removeEventListener('scroll', onScroll); }
    });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) stop(); else if (visible) start();
    });

    replay.addEventListener('click', function () {
      setStage(0);
      var i = 0;
      var timer = setInterval(function () {
        i += 1;
        setStage(i);
        if (i >= STAGES.length - 1) clearInterval(timer);
      }, 1400);
    });

    if (DEMO.prefersReducedMotion()) setStage(STAGES.length - 1, true);
    else setStage(0);

    return { el: root, setStage: setStage };
  }

  DEMO.createTunnel = createTunnel;
})(window.DEMO);
