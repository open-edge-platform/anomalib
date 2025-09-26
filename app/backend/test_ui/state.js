// reactive state store
export const state = {
  projectId: null,
  pipeline: null,
  models: [],
  sources: [],
  sinks: [],
};

const subs = {};
export function subscribe(key, fn) {
  (subs[key] ||= []).push(fn);
}
function emit(key) {
  (subs[key] || []).forEach((fn) => fn(state[key]));
}
export function setState(partial) {
  Object.entries(partial).forEach(([k, v]) => {
    state[k] = v;
    emit(k);
  });
}
