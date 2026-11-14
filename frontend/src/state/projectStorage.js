const VERSION = 1;
const DB_NAME = 'archigen-workspace';

export class ProjectStorage {
  constructor(indexedDB = globalThis.indexedDB) {
    this.indexedDB = indexedDB;
    this.assetCache = new Map();
  }

  async open() {
    if (!this.indexedDB) throw new Error('Local project storage is unavailable in this browser.');
    if (!this.database) this.database = new Promise((resolve, reject) => {
      const request = this.indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore('projects');
        request.result.createObjectStore('assets');
      };
      request.onsuccess = () => {
        request.result.onversionchange = () => request.result.close();
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('Close other ArchiGen tabs to open the saved project.'));
    });
    return this.database;
  }

  async load() {
    const database = await this.open();
    const transaction = database.transaction(['projects', 'assets'], 'readonly');
    const record = await requestValue(transaction.objectStore('projects').get('current'));
    if (!record) return null;
    if (record.version !== VERSION || !record.state || typeof record.state !== 'object' || Array.isArray(record.state)) {
      throw new Error('The saved project format is incompatible. Reset the project to start again.');
    }
    // Read assets in a separate transaction: decoding may yield between reads.
    const assets = await this.readAssets();
    const decode = async value => {
      if (!value || typeof value !== 'object') return value;
      if (value.__projectAsset) {
        const blob = assets.get(value.__projectAsset);
        if (!blob) throw new Error('An image or model is missing from the saved project.');
        if (value.format === 'file') return new File([blob], value.name, { type: blob.type, lastModified: value.lastModified });
        if (value.format === 'blob') return blob;
        return blobToDataURL(blob);
      }
      if (Array.isArray(value)) return Promise.all(value.map(decode));
      return Object.fromEntries(await Promise.all(Object.entries(value).map(async ([key, item]) => [key, await decode(item)])));
    };
    const state = await decode(record.state);
    if (state['app.layout'] && (!Array.isArray(state['app.layout'].walls) || !Array.isArray(state['app.layout'].rooms))) {
      throw new Error('The saved floorplan is invalid. Reset the project to start again.');
    }
    return state;
  }

  async readAssets() {
    const database = await this.open();
    const transaction = database.transaction('assets', 'readonly');
    const store = transaction.objectStore('assets');
    const [keys, values] = await Promise.all([requestValue(store.getAllKeys()), requestValue(store.getAll())]);
    return new Map(keys.map((key, index) => [key, values[index]]));
  }

  async save(state) {
    const assets = new Map();
    const usedCache = new Set();
    const encode = async value => {
      if (value instanceof Blob || (typeof value === 'string' && value.startsWith('data:'))) {
        usedCache.add(value);
        let asset = this.assetCache.get(value);
        if (!asset) {
          const blob = value instanceof Blob ? value : dataURLToBlob(value);
          asset = { id: crypto.randomUUID(), blob };
          this.assetCache.set(value, asset);
        }
        assets.set(asset.id, asset.blob);
        const file = typeof File !== 'undefined' && value instanceof File;
        return { __projectAsset: asset.id, format: file ? 'file' : value instanceof Blob ? 'blob' : 'dataurl',
          ...(file ? { name: value.name, lastModified: value.lastModified } : {}) };
      }
      if (!value || typeof value !== 'object') return value;
      if (Array.isArray(value)) return Promise.all(value.map(encode));
      return Object.fromEntries(await Promise.all(Object.entries(value).map(async ([key, item]) => [key, await encode(item)])));
    };
    const encoded = await encode(state);
    const database = await this.open();
    const transaction = database.transaction(['projects', 'assets'], 'readwrite');
    const done = transactionDone(transaction);
    const store = transaction.objectStore('assets');
    const keys = await requestValue(store.getAllKeys());
    for (const [id, blob] of assets) if (!keys.includes(id)) store.put(blob, id);
    for (const id of keys) if (!assets.has(id)) store.delete(id);
    transaction.objectStore('projects').put({ version: VERSION, state: encoded, savedAt: Date.now() }, 'current');
    await done;
    for (const key of this.assetCache.keys()) if (!usedCache.has(key)) this.assetCache.delete(key);
  }

  async clear() {
    const database = await this.open();
    const transaction = database.transaction(['projects', 'assets'], 'readwrite');
    const done = transactionDone(transaction);
    transaction.objectStore('projects').clear();
    transaction.objectStore('assets').clear();
    await done;
    this.assetCache.clear();
  }
}

export function dataURLToBlob(value) {
  const comma = value.indexOf(',');
  const header = value.slice(0, comma);
  const contents = value.slice(comma + 1);
  const bytes = header.includes(';base64') ? Uint8Array.from(atob(contents), character => character.charCodeAt(0))
    : new TextEncoder().encode(decodeURIComponent(contents));
  return new Blob([bytes], { type: header.slice(5).split(';')[0] });
}

export async function blobToDataURL(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let start = 0; start < bytes.length; start += 32768) binary += String.fromCharCode(...bytes.subarray(start, start + 32768));
  return `data:${blob.type};base64,${btoa(binary)}`;
}

function requestValue(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error || new Error('Project save was interrupted.'));
    transaction.onerror = () => reject(transaction.error);
  });
}
