/* The spine of the demo: site metadata, the shared class palette, and the
 * ordered list of sections.
 *
 * Adding a section is an edit here, as long as its `type` already exists in
 * assets/js/sections.js SECTION_RENDERERS. Adding a model to an existing
 * section is an edit here plus one entry in metrics.js.
 *
 * `classes` is the single source of the class palette for the whole page --
 * table swatches, player legends and the 3D viewer all read it, and the
 * exporter writes it from viz/playback_common.CLASS_COLORS so it can't drift
 * from what the rendered frames actually use.
 */
DEMO.register("manifest", "main", {
  "schemaVersion": 1,
  "generatedAt": "2026-09-19",
  "generatedBy": "hand-written (A1); demo/build_demo.py takes over at B5",

  "site": {
    "navBrand": "sim2real lidar",
    "title": "Sim2Real LiDAR Perception for Autonomous Navigation",
    "subtitle": "Three encoder candidates trained on simulated LiDAR, compared on one shared held-out test set, then run against real Ouster OS1-64 recordings from a Clearpath Jackal.",
    "author": "TODO — your name",
    "degree": "Bachelor Thesis",
    "university": "TODO — university",
    "department": "TODO — department",
    "supervisors": ["TODO — supervisor"],
    "date": "2026",
    "robot": "Clearpath Jackal · Ouster OS1-64",
    "repoUrl": "https://github.com/ahmed-yh/sim2reallidar",
    "disclaimer": "Every result on this page was rendered offline by the project's own scripts and then embedded here. No model runs in your browser. Each playback states the checkpoint it came from."
  },

  "classes": [
    { "id": 0, "key": "environment", "name": "Environment", "color": "#11151a", "object": false },
    { "id": 4, "key": "sphere",      "name": "Sphere",      "color": "#f2a65a", "object": true },
    { "id": 5, "key": "cylinder",    "name": "Cylinder",    "color": "#4fd1c5", "object": true },
    { "id": 6, "key": "box2",        "name": "Box 2",       "color": "#f0d65a", "object": true },
    { "id": 7, "key": "box1",        "name": "Box 1",       "color": "#eb6e5a", "object": true }
  ],

  "sections": [
    {
      "id": "title",
      "type": "hero",
      "nav": false
    },

    {
      "id": "tunnel",
      "type": "tunnel",
      "nav": "The tunnel",
      "title": "Where this happens",
      "blurb": "<p>The environment is a real underground corridor, surveyed with the same LiDAR that later runs the perception models. A SLAM point cloud of the drive was reconstructed into a watertight surface, and that mesh became the simulated world the encoders were trained in — which is what makes a sim-to-real comparison meaningful rather than a comparison against an invented room.</p><p>Both stages below are pre-computed. The surface reconstruction is a CloudCompare result, replayed here as a visualisation, not recomputed in your browser.</p>",
      "payloadId": "tunnel",
      "src": "tunnel/tunnel.js"
    },

    {
      "id": "problem",
      "type": "prose",
      "nav": "Problem",
      "title": "Compressing a LiDAR scan into something a policy can use",
      "html": "<p>A navigation policy cannot consume 65,536 raw LiDAR returns per scan. It needs a compact latent vector. The question this thesis asks is which encoder architecture produces the best one — and, critically, whether an encoder trained entirely in simulation still behaves sensibly on a real sensor.</p><p>Three candidates were trained on the same 7,600-scan simulated dataset and scored on the same held-out split: a point-based <strong>PointNet++</strong>, a range-image <strong>CNN autoencoder</strong>, and <strong>SalsaNext</strong>, a published range-image segmentation network. Two of them also predict a semantic class per point, which is what makes the sim-to-real gap visible rather than just numerical.</p>"
    },

    {
      "id": "candidates",
      "type": "comparison",
      "nav": "Candidates",
      "title": "Three candidates, one shared test set",
      "blurb": "<p>Reconstruction is the one task all three are actually trying to do, so it is the axis they are compared on — in identical normalized units, on identical held-out samples, using the same normalization constant.</p>"
    },

    {
      "id": "perclass",
      "type": "table",
      "nav": "Per-class",
      "title": "Where PointNet++ actually finds the objects",
      "blurb": "<p>Overall point accuracy is a misleading headline here: about 99% of every scan is environment, so a model that predicted \"environment\" everywhere would still score ~99%. The per-class numbers are the real signal.</p>",
      "path": "perClass"
    },

    {
      "id": "sim_playback",
      "type": "tabs",
      "nav": "Simulated drive",
      "title": "Watching it run, frame by frame",
      "blurb": "<p>One held-out drive through the simulated corridor, played at the recording's real rate — and the <em>same</em> drive through all three candidates, so the tabs are a like-for-like comparison rather than three different scenes. The top two rows are the measured range image and the encoder's reconstruction of it; where a candidate has a classifier, the bottom two rows are the ground-truth semantic labels and its predictions. Every frame states which split it belongs to, so you can watch it behave the same on data it never trained on.</p><p>Note the row counts differ: the CNN autoencoder has no classification head at all, so it has nothing to put in a prediction row. That is the candidate's actual capability, not a gap in the recording.</p>",
      "tabs": [
        {
          "label": "PointNet++",
          "type": "player",
          "payloadId": "pointnet2_sim",
          "src": "players/pointnet2_sim.js"
        },
        {
          "label": "CNN autoencoder",
          "type": "player",
          "payloadId": "kevin_sim",
          "src": "players/kevin_sim.js"
        },
        {
          "label": "SalsaNext",
          "type": "player",
          "payloadId": "salsanext_sim",
          "src": "players/salsanext_sim.js"
        }
      ]
    },

    {
      "id": "sim2real",
      "type": "tabs",
      "nav": "Sim2real",
      "title": "The same encoder, on a real sensor",
      "blurb": "<p>Trained entirely in simulation, then handed scans from a real Ouster OS1-64 on a Clearpath Jackal. Nothing was retrained or adapted for this.</p><p><strong>There is no ground truth here.</strong> Real LiDAR carries no semantic labels, so the class row is the model's own prediction with nothing to check it against — which is exactly why the honest claim from this section is qualitative. Predictions are shown at confidence ≥ 0.97, a display threshold chosen by sweeping against the labelled <em>simulated</em> test set; below it a point renders as environment.</p>",
      "tabs": [
        {
          "label": "Video",
          "type": "video",
          "src": "media/bag3_pointnet2.mp4",
          "poster": "media/bag3_pointnet2.poster.jpg",
          "caption": "bag3_20260910_152806 · real Ouster OS1-64 · PointNet++ · played at the bag's own scan rate"
        },
        {
          "label": "Frame player",
          "type": "player",
          "payloadId": "bag3_pointnet2",
          "src": "players/bag3_pointnet2.js"
        }
      ]
    },

    {
      "id": "cloud3d",
      "type": "pointcloud",
      "nav": "3D viewer",
      "title": "What the 256-d latent actually keeps",
      "blurb": "<p>The reconstruction decoder sees the latent vector and nothing else — no skip connections back to the input. So the red points here are decoded from 256 numbers alone, and how closely they trace the corridor walls and the object is a direct, visual read of what survived the bottleneck.</p><p>Drag to orbit, scroll to zoom. Four test-split scans, one per object class — each the scan containing the most points of that class.</p>",
      "payloadId": "pointnet2_recon",
      "src": "pointclouds/pointnet2_recon.js"
    },

    {
      "id": "figures",
      "type": "gallery",
      "nav": "Per-object figures",
      "title": "One scan per object, examined closely",
      "blurb": "<p>The same four test-split scans as the 3D viewer, as static figures. The left pane is the raw range image — the 64×1024 grid the sensor actually produces, with the object visible as a coherent blob against the corridor walls. The right pane is the reconstruction overlay used throughout the evaluation.</p><p>These are the figures behind the per-class table above: each one is the held-out scan containing the most points of its class.</p>",
      "items": [
        {
          "classId": 4,
          "caption": "Sphere · scan 2026_08_25_17_18_39_769 · test split",
          "panes": [
            { "label": "Range image (measured)", "src": "gallery/range_heatmaps/2026_08_25_17_18_39_769_sphere.png" },
            { "label": "Reconstruction overlay", "src": "gallery/recon_viz/2026_08_25_17_18_39_769_sphere.png" }
          ]
        },
        {
          "classId": 5,
          "caption": "Cylinder · scan 2026_08_25_17_18_33_379 · test split",
          "panes": [
            { "label": "Range image (measured)", "src": "gallery/range_heatmaps/2026_08_25_17_18_33_379_cylinder.png" },
            { "label": "Reconstruction overlay", "src": "gallery/recon_viz/2026_08_25_17_18_33_379_cylinder.png" }
          ]
        },
        {
          "classId": 6,
          "caption": "Box 2 · scan 2026_08_25_17_17_58_789 · test split",
          "panes": [
            { "label": "Range image (measured)", "src": "gallery/range_heatmaps/2026_08_25_17_17_58_789_box2.png" },
            { "label": "Reconstruction overlay", "src": "gallery/recon_viz/2026_08_25_17_17_58_789_box2.png" }
          ]
        },
        {
          "classId": 7,
          "caption": "Box 1 · scan 2026_08_25_17_18_24_572 · test split",
          "panes": [
            { "label": "Range image (measured)", "src": "gallery/range_heatmaps/2026_08_25_17_18_24_572_box1.png" },
            { "label": "Reconstruction overlay", "src": "gallery/recon_viz/2026_08_25_17_18_24_572_box1.png" }
          ]
        }
      ]
    }
  ]
});
