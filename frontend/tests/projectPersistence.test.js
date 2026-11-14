import test from 'node:test';
import assert from 'node:assert/strict';
import { IDBFactory } from 'fake-indexeddb';
import { ProjectStorage } from '../src/state/projectStorage.js';
import { createProjectStore } from '../src/state/projectStore.js';
import { generateFurniture } from '../src/state/projectActions.js';

const model = { id: 'chair-1', status: 'success', model_glb_base64: 'data:model/gltf-binary;base64,Z2xURg==', preview_image_base64: 'data:image/png;base64,cG5n' };
const settings = { image: 'data:image/png;base64,cG5n', name: 'Chair', category: 'chair', height: .85, removeBg: true };
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const memoryStorage = () => ({ state: null, async load() { return this.state; }, async save(state) { this.state = state; }, async clear() { this.state = null; } });

test('IndexedDB stores binary assets separately, preserving files and workspace data', async () => {
  const indexedDB = new IDBFactory();
  const storage = new ProjectStorage(indexedDB);
  assert.equal(await storage.load(), null);
  await storage.save({
    'app.file': new File(['floorplan'], 'plan.png', { type: 'image/png', lastModified: 7 }),
    'app.layout': { walls: [], rooms: [], metadata: { source: 'edited' } },
    'studio.generatedModel': model, 'app.facadeDraft': { texture_base64: model.preview_image_base64 },
    'viewer.showRoof': true, 'catalog.room1:item1.colors': { wood: '#112233' },
  });
  const database = await storage.open();
  const record = await new Promise(resolve => {
    const request = database.transaction('projects').objectStore('projects').get('current');
    request.onsuccess = () => resolve(request.result);
  });
  assert.ok(!JSON.stringify(record).includes('data:model/'));
  assert.equal((await storage.readAssets()).size, 3);
  const restored = await new ProjectStorage(indexedDB).load();
  assert.deepEqual(restored['studio.generatedModel'], model);
  assert.equal(restored['app.file'].name, 'plan.png');
  assert.equal(await restored['app.file'].text(), 'floorplan');
  assert.deepEqual(restored['catalog.room1:item1.colors'], { wood: '#112233' });
  await storage.save({ 'studio.generatedModel': model });
  assert.equal((await storage.readAssets()).size, 2);
});

test('restoration recreates uploaded image URLs and explicit showcase navigation wins', async () => {
  const storage = new ProjectStorage(new IDBFactory());
  const first = createProjectStore(storage);
  await first.hydrate('#studio');
  first.set('app.file', new File(['image'], 'image.png', { type: 'image/png' }));
  first.set('app.imageUrl', 'blob:old-session');
  first.set('app.pageView', 'generative_studio');
  await first.flush();
  const restored = createProjectStore(storage);
  await restored.hydrate('#showcase');
  assert.match(restored.read('app.imageUrl'), /^blob:/);
  assert.notEqual(restored.read('app.imageUrl'), 'blob:old-session');
  assert.equal(restored.read('app.currentView'), 'showcase');
  assert.equal(restored.read('app.pageView'), 'generative_studio');
  await restored.reset();
  assert.equal(await storage.load(), null);
});

test('incompatible saves warn and are not overwritten until an explicit reset', async () => {
  const storage = new ProjectStorage(new IDBFactory());
  const database = await storage.open();
  await new Promise(resolve => {
    const transaction = database.transaction('projects', 'readwrite');
    transaction.objectStore('projects').put({ version: 999, state: {} }, 'current');
    transaction.oncomplete = resolve;
  });
  const store = createProjectStore(storage);
  await store.hydrate('#studio');
  assert.match(store.read('$storageError'), /incompatible/);
  store.set('studio.generatedModel', model);
  await store.flush();
  await assert.rejects(storage.load(), /incompatible/);
  await store.reset();
  store.set('studio.generatedModel', model);
  await store.flush();
  assert.deepEqual((await storage.load())['studio.generatedModel'], model);
});

test('ordered writes and reset cannot be undone by old saves or callbacks', async () => {
  const gate = deferred();
  const storage = memoryStorage();
  const originalSave = storage.save.bind(storage);
  let count = 0;
  storage.save = async state => { if (++count === 1) await gate.promise; await originalSave(state); };
  const store = createProjectStore(storage);
  await store.hydrate();
  const oldEpoch = store.epoch;
  store.set('app.layout', { revision: 1 });
  const reset = store.reset();
  store.set('app.layout', { revision: 2 });
  store.set('app.layout', { revision: 99 }, null, oldEpoch);
  gate.resolve();
  await reset;
  await store.flush();
  assert.equal(storage.state['app.layout'].revision, 2);
});

test('save failure leaves the live model intact and shows a warning', async () => {
  const storage = memoryStorage();
  storage.save = async () => { throw new Error('Storage quota exceeded'); };
  const store = createProjectStore(storage);
  await store.hydrate();
  store.set('studio.generatedModel', model);
  await store.flush();
  assert.deepEqual(store.read('studio.generatedModel'), model);
  assert.match(store.read('$storageError'), /quota/);
});

test('generation survives navigation and completes in the shared project', async () => {
  const store = createProjectStore(memoryStorage());
  await store.hydrate();
  const response = deferred();
  const generation = generateFurniture(store, settings, { post: () => response.promise });
  store.set('app.pageView', 'floorplan');
  assert.equal(store.read('request.furniture'), true);
  response.resolve({ data: model });
  await generation;
  assert.equal(store.read('studio.generatedModel').id, model.id);
  assert.equal(store.read('request.furniture'), false);
});

test('new input, newer request and reset each invalidate old generation responses', async () => {
  for (const action of ['input', 'newer', 'reset']) {
    const store = createProjectStore(memoryStorage());
    await store.hydrate();
    const response = deferred();
    const generation = generateFurniture(store, settings, { post: () => response.promise });
    if (action === 'input') store.set('studio.furnitureImageBase64', 'new-image');
    if (action === 'newer') await generateFurniture(store, settings, { post: async () => ({ data: { ...model, id: 'newer' } }) });
    if (action === 'reset') await store.reset();
    response.resolve({ data: model });
    await generation;
    assert.equal(store.read('studio.generatedModel', null)?.id, action === 'newer' ? 'newer' : undefined);
  }
});

test('reload retains last completed model and signals interruption without retrying', async () => {
  const storage = memoryStorage();
  const store = createProjectStore(storage);
  await store.hydrate();
  store.set('studio.generatedModel', model);
  store.startRequest('furniture');
  await store.flush();
  const restored = createProjectStore(storage);
  await restored.hydrate('#studio');
  assert.deepEqual(restored.read('studio.generatedModel'), model);
  assert.equal(restored.read('request.furniture'), false);
  assert.match(restored.read('$notice'), /interrupted/);
});
