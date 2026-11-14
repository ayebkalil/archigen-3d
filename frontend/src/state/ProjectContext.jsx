import React, { createContext, useCallback, useContext, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { ProjectStorage } from './projectStorage.js';
import { createProjectStore } from './projectStore.js';

const ProjectContext = createContext(null);

export function ProjectProvider({ children }) {
  const [store] = useState(() => createProjectStore(new ProjectStorage()));
  const [ready, setReady] = useState(false);
  const epoch = useSyncExternalStore(store.subscribe, () => store.epoch);
  useEffect(() => {
    let active = true;
    store.hydrate(window.location.hash).then(() => { if (active) setReady(true); });
    const flush = () => store.flush();
    window.addEventListener('pagehide', flush);
    document.addEventListener('visibilitychange', flush);
    return () => {
      active = false;
      window.removeEventListener('pagehide', flush);
      document.removeEventListener('visibilitychange', flush);
    };
  }, [store]);
  return <ProjectContext.Provider value={store}>
    {ready ? <React.Fragment key={epoch}>{children}<ProjectStatus /></React.Fragment>
      : <div className="p-8 font-mono" role="status">Restoring your project…</div>}
  </ProjectContext.Provider>;
}

export function useProject() { return useContext(ProjectContext); }

export function useProjectState(key, initial) {
  const store = useProject();
  const initialRef = useRef();
  if (!initialRef.current) initialRef.current = { value: typeof initial === 'function' ? initial() : initial };
  const fallback = initialRef.current.value;
  const epoch = store.epoch;
  const snapshot = useCallback(() => store.read(key, fallback), [store, key, fallback]);
  const value = useSyncExternalStore(store.subscribe, snapshot);
  const setter = useCallback(next => store.set(key, next, fallback, epoch), [store, key, fallback, epoch]);
  return [value, setter];
}

function ProjectStatus() {
  const [error, setError] = useProjectState('$storageError', null);
  const [notice, setNotice] = useProjectState('$notice', null);
  if (!error && !notice) return null;
  return <div role="alert" className="fixed bottom-4 left-4 right-4 z-[100] rounded-xl border border-orange-300 bg-orange-50 p-4 text-sm text-orange-950 shadow-lg">
    {error || notice}
    <button className="ml-4 underline" onClick={() => error ? setError(null) : setNotice(null)}>Dismiss</button>
  </div>;
}
