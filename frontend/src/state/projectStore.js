export function createProjectStore(storage) {
  let state = {};
  let epoch = 0;
  let queue = Promise.resolve();
  let hydration;
  let blocked = false;
  let timer;
  const listeners = new Set();
  const requests = new Map();
  const urls = new Set();
  const emit = () => listeners.forEach(listener => listener());
  const warning = error => {
    state = { ...state, $storageError: `Project could not be saved or restored: ${error.message}. Your current work stays in this tab.` };
    emit();
  };
  const snapshot = () => {
    const next = Object.fromEntries(Object.entries(state).filter(([key]) => !key.startsWith('$')));
    if (next['app.imageUrl']?.startsWith('blob:')) next['app.imageUrl'] = null;
    return next;
  };
  const store = {
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    get epoch() { return epoch; },
    read(key, fallback) { return Object.hasOwn(state, key) ? state[key] : fallback; },
    set(key, value, fallback, expectedEpoch = epoch) {
      if (epoch !== expectedEpoch) return;
      const previous = store.read(key, fallback);
      const next = typeof value === 'function' ? value(previous) : value;
      if (Object.is(previous, next)) return;
      state = { ...state, [key]: next };
      // A change of generation inputs invalidates outstanding responses.
      if (['studio.furnitureImageBase64', 'studio.furnitureName', 'studio.furnitureCategory', 'studio.targetHeightM', 'studio.removeBg'].includes(key)) {
        store.invalidate('furniture');
      }
      if (['app.selectedStyle', 'app.frontWallId', 'app.numFloors', 'app.layout'].includes(key)) store.invalidate('facade');
      if (['app.file', 'app.layout'].includes(key)) store.invalidate('design');
      if (['app.file', 'app.layout'].includes(key)) store.invalidate('design');
      emit();
      if (!key.startsWith('$')) {
        const largeResult = /generatedModel|facadeDraft|layout|file|furnishingData|request\./.test(key);
        if (largeResult) store.flush();
        else { clearTimeout(timer); timer = setTimeout(() => store.flush(), 200); }
      }
    },
    async hydrate(hash = '') {
      if (!hydration) hydration = (async () => {
        try {
          state = await storage.load() || {};
          if (state['app.file'] instanceof Blob && !state['app.imageUrl']) {
            const url = URL.createObjectURL(state['app.file']);
            urls.add(url);
            state['app.imageUrl'] = url;
          }
          if (Object.entries(state).some(([key, value]) => key.startsWith('request.') && value)) {
            state.$notice = 'A generation was interrupted by reloading. Your last completed result has been restored; generate again when ready.';
          }
          for (const key of Object.keys(state)) if (key.startsWith('request.')) state[key] = false;
        } catch (error) { blocked = true; state = {}; warning(error); }
        if (hash === '#showcase') state['app.currentView'] = 'showcase';
        if (hash === '#studio') state['app.currentView'] = 'studio';
        emit();
      })();
      return hydration;
    },
    flush() {
      clearTimeout(timer);
      if (blocked) return queue;
      const saved = snapshot();
      queue = queue.then(() => storage.save(saved)).catch(warning);
      return queue;
    },
    async reset() {
      clearTimeout(timer);
      epoch++;
      requests.clear();
      for (const url of urls) URL.revokeObjectURL(url);
      urls.clear();
      state = { 'app.currentView': 'studio' };
      blocked = false;
      // Queue the deletion behind any writes already in flight.
      queue = queue.then(() => storage.clear()).catch(warning);
      emit();
      await queue;
    },
    startRequest(channel) {
      const token = { epoch, id: Symbol(channel) };
      requests.set(channel, token);
      store.set(`request.${channel}`, true);
      return token;
    },
    isCurrent(channel, token) { return token.epoch === epoch && requests.get(channel) === token; },
    finishRequest(channel, token) {
      if (!store.isCurrent(channel, token)) return false;
      requests.delete(channel);
      store.set(`request.${channel}`, false);
      return true;
    },
    invalidate(channel) {
      requests.delete(channel);
      if (state[`request.${channel}`]) state = { ...state, [`request.${channel}`]: false };
    },
    trackURL(url) { urls.add(url); return url; },
    releaseURL(url) { URL.revokeObjectURL(url); urls.delete(url); },
    clearSection(prefix) {
      state = Object.fromEntries(Object.entries(state).filter(([key]) => !key.startsWith(prefix)));
      emit();
      store.flush();
    },
  };
  return store;
}
