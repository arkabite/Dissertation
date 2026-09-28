import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const LEVEL_COLORS = { Low: 0x4c78a8, Medium: 0xb0b0b0, High: 0xe45756 };
const MESH_URL = "/fsaverage_brain.glb";

export default function Brain3D({ points }) {
  const rootRef = useRef(null);
  const canvasRef = useRef(null);
  const [tooltip, setTooltip] = useState(null);
  const [meshError, setMeshError] = useState(false);

  useEffect(() => {
    const root = rootRef.current;
    const canvasHost = canvasRef.current;
    if (!root || !canvasHost) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0.55, -2.4);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    canvasHost.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0.15, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.55);
    keyLight.position.set(2, 3, -3);
    scene.add(keyLight);
    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-2, -1, 3);
    scene.add(backLight);

    const brainMaterial = new THREE.MeshPhongMaterial({
      color: 0xd9c3bd,
      specular: 0x222222,
      shininess: 12,
      side: THREE.DoubleSide,
    });
    const scalp = new THREE.Mesh(
      new THREE.SphereGeometry(1, 48, 32),
      new THREE.MeshPhongMaterial({
        color: 0xe7bda7,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    scene.add(scalp);

    const markers = [];
    (points || []).forEach((point) => {
      if (point.x3d == null || point.y3d == null || point.z3d == null) return;
      const isHbO = point.signal === "HbO";
      const geometry = isHbO
        ? new THREE.SphereGeometry(0.038, 14, 14)
        : new THREE.BoxGeometry(0.06, 0.06, 0.06);
      const marker = new THREE.Mesh(
        geometry,
        new THREE.MeshPhongMaterial({ color: LEVEL_COLORS[point.level] || 0x999999 })
      );
      const scale = isHbO ? 0.975 : 1.025;
      marker.position.set(point.x3d * scale, point.z3d * scale, -point.y3d * scale);
      marker.userData = point;
      scene.add(marker);
      markers.push(marker);
    });

    const loader = new GLTFLoader();
    let disposed = false;
    loader.load(
      MESH_URL,
      (gltf) => {
        if (disposed) return;
        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.material = brainMaterial;
            child.geometry.computeVertexNormals();
          }
        });
        scene.add(gltf.scene);
      },
      undefined,
      () => {
        if (!disposed) setMeshError(true);
      }
    );

    const resize = () => {
      const width = Math.max(1, canvasHost.clientWidth);
      const height = Math.max(240, Math.min(width, 520));
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvasHost);
    resize();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPointerMove = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(markers)[0];
      if (!hit) {
        setTooltip(null);
        return;
      }
      setTooltip({
        left: event.clientX - rect.left + 12,
        top: event.clientY - rect.top + 12,
        point: hit.object.userData,
      });
    };
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", () => setTooltip(null));
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      controls.dispose();
      scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material && object.material !== brainMaterial) {
          object.material.dispose();
        }
      });
      brainMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [points]);

  return (
    <div className="brain3d" ref={rootRef}>
      <div className="brain3d-canvas" ref={canvasRef} />
      {tooltip && (
        <div className="brain3d-tooltip" style={{ left: tooltip.left, top: tooltip.top }}>
          <strong>{tooltip.point.bare_channel}</strong> ({tooltip.point.signal})
          <br />Level: {tooltip.point.level}
          <br />Region: {tooltip.point.functional_region}
          <br />Rule {tooltip.point.rule_id} -&gt; {tooltip.point.consequent}
          <br />ACC={tooltip.point.accuracy != null ? Number(tooltip.point.accuracy).toFixed(2) : "n/a"}
        </div>
      )}
      {meshError && (
        <p className="brain3d-status">
          Brain surface unavailable; showing sensor positions only. Ensure
          <code>public/fsaverage_brain.glb</code> exists.
        </p>
      )}
      <p className="caption">Drag to rotate, scroll to zoom. Markers without 3D coordinates are omitted.</p>
    </div>
  );
}
