// Exterior-wall facade payloads exactly as the studio builds them (App.facadePayloadFor),
// using the frontend's own opening fitting. Used by scripts/evaluate_facade_engines.py.
//   node scripts/facade_payloads.mjs < {"kind":"analysis","response":{...},"scale":0.03,"num_floors":1}
//   node scripts/facade_payloads.mjs < {"kind":"sample","scale":0.03}
import { buildingLevels, fitHouseOpenings } from '../frontend/src/geometry/houseGeometry.js';
import { createSampleHouse } from '../frontend/src/geometry/sampleHousePlan.js';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));

let layout;
if (input.kind === 'sample') {
  layout = createSampleHouse(input.scale ?? 0.03);
} else {
  const data = input.response;
  // Same reconstruction as App.executeAnalysis.
  layout = {
    metadata: { canvas_size: [512, 512], scale_meters_per_pixel: input.scale },
    walls: data.walls.map(({ facade_texture, ...wall }) => wall),
    rooms: data.rooms.map((r, i) => ({ id: `room_${i}`, room_type: r.room_type, surface_m2: r.surface_m2, polygon_meters: r.polygon_points })),
    doors: data.doors || [], windows: data.windows || [], envelope: data.envelope || [],
  };
}
const numFloors = input.num_floors ?? 1;
const payloads = [];
for (const target of layout.walls.filter(wall => wall.is_exterior)) {
  const level = buildingLevels(layout, numFloors).find(item => item.walls.includes(target));
  if (!level) continue;
  const openings = fitHouseOpenings(level, layout.metadata.scale_meters_per_pixel)
    .filter(opening => opening.associated_wall === target.id);
  const { facade_texture: _ignored, ...wall } = target;
  payloads.push({ wall: { ...wall, height_m: level.unitHeight, openings }, wall_id: target.id,
    num_floors: target.level_id ? 1 : numFloors, level_id: target.level_id ?? null });
}
process.stdout.write(JSON.stringify(payloads));
