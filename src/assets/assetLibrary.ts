import * as THREE from 'three';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { createAutonomyVehicle, type AutonomyVehicleKind, type AutonomyVehicleStyle } from '../visual/autonomyVehicle';
import { setPresentationLayer } from '../visual/layers';

export type VehicleAssetKey = 'sedan' | 'suv' | 'van' | 'truck' | 'delivery' | 'police';
export type PropAssetKey =
  | 'cone'
  | 'box'
  | 'construction-barrier'
  | 'construction-cone'
  | 'construction-light'
  | 'light-curved'
  | 'light-curved-cross'
  | 'light-square'
  | 'light-square-cross'
  | 'road-bend'
  | 'road-curve'
  | 'sign-highway'
  | 'road-straight';

type AssetKey = VehicleAssetKey | PropAssetKey;
type QuaterniusVehicleKey = 'q-normal-1' | 'q-normal-2' | 'q-suv' | 'q-taxi' | 'q-cop';
type CachedAssetKey = AssetKey | 'premium-car' | QuaterniusVehicleKey;

type StyleKind = 'ego' | 'traffic' | 'service' | 'prop' | 'warning' | 'street';

const VEHICLE_PATHS: Record<VehicleAssetKey, string> = {
  sedan: '/assets/kenney/vehicles/sedan.glb',
  suv: '/assets/kenney/vehicles/suv.glb',
  van: '/assets/kenney/vehicles/van.glb',
  truck: '/assets/kenney/vehicles/truck.glb',
  delivery: '/assets/kenney/vehicles/delivery.glb',
  police: '/assets/kenney/vehicles/police.glb',
};

const PROP_PATHS: Record<PropAssetKey, string> = {
  cone: '/assets/kenney/vehicles/cone.glb',
  box: '/assets/kenney/vehicles/box.glb',
  'construction-barrier': '/assets/kenney/props/construction-barrier.glb',
  'construction-cone': '/assets/kenney/props/construction-cone.glb',
  'construction-light': '/assets/kenney/props/construction-light.glb',
  'light-curved': '/assets/kenney/props/light-curved.glb',
  'light-curved-cross': '/assets/kenney/props/light-curved-cross.glb',
  'light-square': '/assets/kenney/props/light-square.glb',
  'light-square-cross': '/assets/kenney/props/light-square-cross.glb',
  'road-bend': '/assets/kenney/props/road-bend.glb',
  'road-curve': '/assets/kenney/props/road-curve.glb',
  'sign-highway': '/assets/kenney/props/sign-highway.glb',
  'road-straight': '/assets/kenney/props/road-straight.glb',
};

const PREMIUM_VEHICLE_PATH = '/assets/vehicles/threejs/ferrari.glb';

const QUATERNIUS_VEHICLE_PATHS: Record<QuaterniusVehicleKey, string> = {
  'q-normal-1': '/assets/vehicles/quaternius/normal-car-1.obj',
  'q-normal-2': '/assets/vehicles/quaternius/normal-car-2.obj',
  'q-suv': '/assets/vehicles/quaternius/suv.obj',
  'q-taxi': '/assets/vehicles/quaternius/taxi.obj',
  'q-cop': '/assets/vehicles/quaternius/cop.obj',
};

const PALETTE = {
  egoBody: 0xffffff,
  trafficBody: 0xf3f6f9,
  serviceBody: 0xe7edf3,
  glass: 0x17212b,
  tire: 0x111827,
  trim: 0xcfd8e3,
  prop: 0xe7edf3,
  warning: 0xff7a2f,
  street: 0xd4dce5,
  cyan: 0x27c8ff,
};

export class AssetLibrary {
  private readonly dracoLoader = new DRACOLoader();
  private readonly loader = new GLTFLoader();
  private readonly objLoader = new OBJLoader();
  private readonly cache = new Map<CachedAssetKey, THREE.Group>();
  private trafficVariantIndex = 0;

  constructor() {
    this.dracoLoader.setDecoderPath('/assets/draco/');
    this.loader.setDRACOLoader(this.dracoLoader);
  }

  async loadAll(): Promise<void> {
    const entries = [
      ...Object.entries(VEHICLE_PATHS),
      ...Object.entries(PROP_PATHS),
    ] as [CachedAssetKey, string][];
    const quaterniusEntries = Object.entries(QUATERNIUS_VEHICLE_PATHS) as [QuaterniusVehicleKey, string][];
    await Promise.all([
      ...entries.map(async ([key, path]) => {
        try {
          const gltf = await this.loader.loadAsync(path);
          this.cache.set(key, gltf.scene);
        } catch (error) {
          console.warn(`Could not load asset ${key} from ${path}`, error);
        }
      }),
      ...quaterniusEntries.map(async ([key, path]) => {
        try {
          const obj = await this.objLoader.loadAsync(path);
          this.cache.set(key, obj);
        } catch (error) {
          console.warn(`Could not load vehicle asset ${key} from ${path}`, error);
        }
      }),
    ]);
  }

  createVehicle(key: VehicleAssetKey, style: StyleKind = 'traffic'): THREE.Group | null {
    const quaternius = this.createQuaterniusVehicle(key, style);
    if (quaternius) return quaternius;

    const source = this.cache.get(key);
    if (source) {
      const model = source.clone(true);
      this.restyle(model, style);
      this.normalize(model, this.targetSizeForVehicle(key));
      this.addTrackingOverlay(model, this.targetSizeForVehicle(key), style);
      return model;
    }

    const fallback = createAutonomyVehicle(this.autonomyVehicleKind(key), this.autonomyVehicleStyle(style));
    this.addTrackingOverlay(fallback, this.targetSizeForVehicle(key), style);
    return fallback;
  }

  createProp(key: PropAssetKey, style: StyleKind = 'prop', targetSize?: THREE.Vector3): THREE.Group | null {
    const source = this.cache.get(key);
    if (!source) return null;
    const model = source.clone(true);
    this.restyle(model, style);
    this.normalize(model, targetSize ?? this.targetSizeForProp(key));
    return model;
  }

  private restyle(root: THREE.Object3D, style: StyleKind): void {
    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const key = `${object.name} ${Array.isArray(object.material) ? object.material.map((m) => m.name).join(' ') : object.material.name}`.toLowerCase();
      object.castShadow = false;
      object.receiveShadow = false;
      object.userData.sharedGeometry = true;
      object.material = this.materialFor(key, style);
    });
  }

  private materialFor(key: string, style: StyleKind): THREE.Material {
    if (key.includes('glass') || key.includes('window') || key.includes('windscreen')) {
      return new THREE.MeshStandardMaterial({
        color: PALETTE.glass,
        roughness: 0.22,
        metalness: 0.2,
        transparent: true,
        opacity: 0.78,
      });
    }
    if (key.includes('wheel') || key.includes('tire') || key.includes('tyre')) {
      return new THREE.MeshStandardMaterial({ color: PALETTE.tire, roughness: 0.62 });
    }
    if (style === 'warning') {
      return new THREE.MeshStandardMaterial({ color: PALETTE.warning, roughness: 0.72 });
    }
    if (style === 'street') {
      return new THREE.MeshStandardMaterial({ color: PALETTE.street, roughness: 0.78 });
    }
    if (key.includes('light') || key.includes('lamp')) {
      return new THREE.MeshBasicMaterial({ color: PALETTE.cyan });
    }
    if (style === 'service') {
      return new THREE.MeshStandardMaterial({ color: PALETTE.serviceBody, roughness: 0.48, metalness: 0.05 });
    }
    if (style === 'ego') {
      return new THREE.MeshStandardMaterial({ color: PALETTE.egoBody, roughness: 0.34, metalness: 0.08 });
    }
    if (style === 'traffic') {
      return new THREE.MeshStandardMaterial({
        color: PALETTE.trafficBody,
        roughness: 0.52,
        metalness: 0.04,
        transparent: true,
        opacity: 0.88,
      });
    }
    return new THREE.MeshStandardMaterial({ color: PALETTE.prop, roughness: 0.76 });
  }

  private createQuaterniusVehicle(key: VehicleAssetKey, style: StyleKind): THREE.Group | null {
    const variant = this.selectQuaterniusVehicle(key, style);
    if (!variant) return null;
    const source = this.cache.get(variant);
    if (!source) return null;

    const model = source.clone(true);
    const target = this.targetSizeForQuaterniusVehicle(variant, style);
    this.restyleQuaterniusVehicle(model, variant, style);
    this.normalize(model, target);
    this.addTrackingOverlay(model, target, style);
    return model;
  }

  private selectQuaterniusVehicle(key: VehicleAssetKey, style: StyleKind): QuaterniusVehicleKey | null {
    if (key === 'truck' || key === 'delivery') return null;
    if (style === 'ego') return 'q-normal-2';
    if (key === 'suv') return 'q-suv';
    if (key === 'police') return 'q-cop';
    if (key === 'van') return 'q-taxi';
    const variants: QuaterniusVehicleKey[] = ['q-normal-1', 'q-normal-2', 'q-taxi', 'q-suv'];
    const variant = variants[this.trafficVariantIndex % variants.length];
    this.trafficVariantIndex++;
    return variant;
  }

  private restyleQuaterniusVehicle(root: THREE.Object3D, variant: QuaterniusVehicleKey, style: StyleKind): void {
    const bodyColor = this.bodyColorForQuaterniusVehicle(variant, style);
    const materials = {
      body: new THREE.MeshStandardMaterial({
        color: bodyColor,
        roughness: style === 'ego' ? 0.28 : 0.42,
        metalness: 0.08,
        transparent: style === 'traffic',
        opacity: style === 'traffic' ? 0.9 : 1,
      }),
      glass: new THREE.MeshStandardMaterial({
        color: 0x101820,
        roughness: 0.2,
        metalness: 0.18,
        transparent: true,
        opacity: 0.88,
      }),
      tire: new THREE.MeshStandardMaterial({ color: 0x090d12, roughness: 0.7, metalness: 0.04 }),
      rim: new THREE.MeshStandardMaterial({ color: 0xcdd6df, roughness: 0.35, metalness: 0.34 }),
      headlight: new THREE.MeshBasicMaterial({ color: style === 'ego' ? 0x8beaff : 0xf7fbff }),
      tail: new THREE.MeshBasicMaterial({ color: 0xff3c5c }),
      blue: new THREE.MeshBasicMaterial({ color: 0x2794ff }),
      white: new THREE.MeshStandardMaterial({ color: 0xf8fbff, roughness: 0.38, metalness: 0.08 }),
    };

    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      object.castShadow = false;
      object.receiveShadow = false;
      object.userData.sharedGeometry = true;
      if (Array.isArray(object.material)) {
        object.material = object.material.map((material) => (
          this.materialForQuaterniusPart(`${object.name} ${material.name}`, variant, materials)
        ));
      } else {
        object.material = this.materialForQuaterniusPart(`${object.name} ${object.material.name}`, variant, materials);
      }
    });
  }

  private materialForQuaterniusPart(
    partName: string,
    variant: QuaterniusVehicleKey,
    materials: {
      body: THREE.Material;
      glass: THREE.Material;
      tire: THREE.Material;
      rim: THREE.Material;
      headlight: THREE.Material;
      tail: THREE.Material;
      blue: THREE.Material;
      white: THREE.Material;
    },
  ): THREE.Material {
    const key = partName.toLowerCase();
    if (key.includes('window')) return materials.glass;
    if (key.includes('headlight') || key.includes('whitelight')) return materials.headlight;
    if (key.includes('taillight')) return materials.tail;
    if (key.includes('bluelight')) return materials.blue;
    if (key.includes('wheel') && key.includes('grey')) return materials.rim;
    if (key.includes('wheel') || key.includes('black')) return materials.tire;
    if (key.includes('grey')) return materials.rim;
    if (key.includes('white') && variant === 'q-cop') return materials.white;
    return materials.body;
  }

  private bodyColorForQuaterniusVehicle(variant: QuaterniusVehicleKey, style: StyleKind): number {
    if (style === 'ego') return 0xffffff;
    if (variant === 'q-taxi') return 0xf2c84b;
    if (variant === 'q-cop') return 0x111827;
    if (variant === 'q-suv') return 0xe9eef5;
    return variant === 'q-normal-1' ? 0xdfe8f0 : 0xf5f7fb;
  }

  private createPremiumVehicle(key: VehicleAssetKey, style: StyleKind): THREE.Group | null {
    const source = this.cache.get('premium-car');
    if (!source) return null;
    const model = source.clone(true);
    const target = this.targetSizeForPremiumVehicle(key, style);
    this.restylePremiumVehicle(model, style);
    this.normalize(model, target);
    this.addTrackingOverlay(model, target, style);
    return model;
  }

  private restylePremiumVehicle(root: THREE.Object3D, style: StyleKind): void {
    const bodyColor = style === 'ego' ? 0xffffff : style === 'service' ? 0xe7edf3 : 0xf3f6f9;
    const materials = {
      body: new THREE.MeshPhysicalMaterial({
        color: bodyColor,
        roughness: style === 'ego' ? 0.24 : 0.33,
        metalness: 0.08,
        clearcoat: 0.72,
        clearcoatRoughness: 0.18,
        transparent: style === 'traffic',
        opacity: style === 'traffic' ? 0.86 : 1,
      }),
      glass: new THREE.MeshPhysicalMaterial({
        color: 0x0e141c,
        roughness: 0.16,
        metalness: 0.18,
        clearcoat: 0.45,
        transparent: true,
        opacity: 0.92,
      }),
      tire: new THREE.MeshStandardMaterial({ color: 0x07090d, roughness: 0.72, metalness: 0.03 }),
      rim: new THREE.MeshStandardMaterial({ color: 0xe1e7ee, roughness: 0.24, metalness: 0.72 }),
      interior: new THREE.MeshStandardMaterial({ color: 0x121820, roughness: 0.44, metalness: 0.08 }),
      headlight: new THREE.MeshBasicMaterial({ color: style === 'ego' ? 0x80e7ff : 0xeef8ff }),
      tail: new THREE.MeshBasicMaterial({ color: 0xff3555 }),
    };

    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const materialName = Array.isArray(object.material)
        ? object.material.map((material) => material.name).join(' ')
        : object.material.name;
      const key = `${object.name} ${materialName}`.toLowerCase();
      object.castShadow = false;
      object.receiveShadow = false;
      object.userData.sharedGeometry = true;

      if (key.includes('taillight') || key.includes('lights_red') || key.includes('red_lights')) {
        object.material = materials.tail;
      } else if (key.includes('projector') || key.includes('headlight')) {
        object.material = materials.headlight;
      } else if (key.includes('glass')) {
        object.material = materials.glass;
      } else if (key.includes('tire') || key.includes('tyre')) {
        object.material = materials.tire;
      } else if (key.includes('rim') || key.includes('metal') || key.includes('chrome') || key.includes('wheel')) {
        object.material = key.includes('wheel') ? materials.tire : materials.rim;
      } else if (key.includes('body_color') || key.includes('body')) {
        object.material = materials.body;
      } else {
        object.material = materials.interior;
      }
    });
  }

  private addTrackingOverlay(root: THREE.Group, target: THREE.Vector3, style: StyleKind): void {
    const color = style === 'ego' ? 0x18c7ff : style === 'service' ? 0xffb341 : 0x507cff;
    const opacity = style === 'ego' ? 0.28 : 0.17;
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(target.z * 0.55, target.z * 0.6, 80),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.065;
    setPresentationLayer(ring);
    root.add(ring);

    const box = this.createGroundTrackingBox(target.x + 0.78, target.z + 0.78, color, style === 'ego' ? 0.58 : 0.34);
    box.position.y = 0.08;
    setPresentationLayer(box);
    root.add(box);

    if (style === 'ego') {
      const roofSensor = new THREE.Mesh(
        new THREE.BoxGeometry(target.x * 0.22, 0.055, target.z * 0.11),
        new THREE.MeshBasicMaterial({ color: 0x19d4ff, transparent: true, opacity: 0.9 }),
      );
      roofSensor.position.set(0, target.y + 0.08, -target.z * 0.03);
      setPresentationLayer(roofSensor);
      root.add(roofSensor);
    }
  }

  private createGroundTrackingBox(width: number, length: number, color: number, opacity: number): THREE.Group {
    const group = new THREE.Group();
    const material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    const thickness = 0.05;
    const longGeo = new THREE.BoxGeometry(thickness, 0.025, length);
    const shortGeo = new THREE.BoxGeometry(width, 0.025, thickness);
    for (const x of [-width / 2, width / 2]) {
      const edge = new THREE.Mesh(longGeo, material);
      edge.position.x = x;
      group.add(edge);
    }
    for (const z of [-length / 2, length / 2]) {
      const edge = new THREE.Mesh(shortGeo, material);
      edge.position.z = z;
      group.add(edge);
    }
    return group;
  }

  private normalize(root: THREE.Group, target: THREE.Vector3): void {
    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    if (size.x <= 0 || size.y <= 0 || size.z <= 0) return;
    const scale = Math.min(target.x / size.x, target.y / size.y, target.z / size.z);
    root.scale.multiplyScalar(scale);
    const scaledBox = new THREE.Box3().setFromObject(root);
    const center = scaledBox.getCenter(new THREE.Vector3());
    const minY = scaledBox.min.y;
    root.position.sub(new THREE.Vector3(center.x, minY, center.z));
  }

  private targetSizeForVehicle(key: VehicleAssetKey): THREE.Vector3 {
    if (key === 'truck' || key === 'delivery') return new THREE.Vector3(3.1, 2.1, 7.2);
    if (key === 'van') return new THREE.Vector3(2.6, 1.9, 5.6);
    if (key === 'suv' || key === 'police') return new THREE.Vector3(2.55, 1.65, 5.1);
    return new THREE.Vector3(2.35, 1.45, 4.75);
  }

  private targetSizeForQuaterniusVehicle(variant: QuaterniusVehicleKey, style: StyleKind): THREE.Vector3 {
    if (style === 'ego') return new THREE.Vector3(2.42, 1.42, 5.05);
    if (variant === 'q-suv' || variant === 'q-cop') return new THREE.Vector3(2.55, 1.68, 5.15);
    if (variant === 'q-taxi') return new THREE.Vector3(2.45, 1.48, 5.05);
    return new THREE.Vector3(2.36, 1.42, 4.9);
  }

  private targetSizeForPremiumVehicle(key: VehicleAssetKey, style: StyleKind): THREE.Vector3 {
    if (style === 'ego') return new THREE.Vector3(2.45, 1.35, 5.2);
    if (key === 'van') return new THREE.Vector3(2.5, 1.45, 5.45);
    if (key === 'suv' || key === 'police') return new THREE.Vector3(2.48, 1.42, 5.25);
    return new THREE.Vector3(2.35, 1.32, 5.05);
  }

  private autonomyVehicleKind(key: VehicleAssetKey): AutonomyVehicleKind {
    if (key === 'truck') return 'truck';
    if (key === 'delivery') return 'bus';
    if (key === 'van') return 'van';
    if (key === 'suv' || key === 'police') return 'suv';
    return 'sedan';
  }

  private autonomyVehicleStyle(style: StyleKind): AutonomyVehicleStyle {
    if (style === 'ego') return 'ego';
    if (style === 'service') return 'service';
    return 'traffic';
  }

  private targetSizeForProp(key: PropAssetKey): THREE.Vector3 {
    if (key === 'road-curve' || key === 'road-bend') return new THREE.Vector3(23, 1, 23);
    if (key === 'road-straight') return new THREE.Vector3(23, 1, 23);
    if (key.includes('light')) return new THREE.Vector3(2.8, 5.8, 2.8);
    if (key.includes('barrier')) return new THREE.Vector3(3.8, 1.3, 1);
    if (key.includes('cone')) return new THREE.Vector3(1.1, 1.5, 1.1);
    if (key.includes('sign')) return new THREE.Vector3(4, 4, 1);
    return new THREE.Vector3(2, 2, 2);
  }
}
