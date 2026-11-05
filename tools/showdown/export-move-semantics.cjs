// Sidecar export: never rewrites frozen team catalogs or battle inputs.
const fs = require('node:fs');
const {Dex} = require('pokemon-showdown');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const format = input.format || 'gen9ou';
const dex = Dex.forFormat(format);

function callbacks(value, path = '') {
  if (!value || typeof value !== 'object') return [];
  return Object.entries(value).flatMap(([k, v]) => {
    const p = path ? `${path}.${k}` : k;
    return typeof v === 'function' ? [p] : callbacks(v, p);
  });
}

const moves = {};
for (const id of [...new Set(input.move_ids || [])].sort()) {
  const m = dex.moves.get(id);
  if (!m.exists || m.id !== id) throw new Error(`Unknown or noncanonical move: ${id}`);
  const effects = [];
  function readEffect(obj, recipient, phase, path, chance = null) {
    if (!obj) return;
    for (const [stat, stages] of Object.entries(obj.boosts || {})) {
      effects.push({kind: 'stat_change', recipient, phase, stat, stages,
        source_field: `${path}boosts.${stat}`, chance});
    }
    if (obj.status) effects.push({kind: 'major_status', recipient, phase,
      status: obj.status, source_field: `${path}status`, chance});
  }
  const recipient = m.target === 'self' ? 'self' :
    ['normal', 'adjacentFoe', 'allAdjacentFoes', 'randomNormal'].includes(m.target) ? 'target' : 'other';
  readEffect(m, recipient, 'primary', '');
  readEffect(m.self, 'self', 'primary', 'self.');
  // Dex normalizes secondary into secondaries; read only the normalized array.
  (m.secondaries || []).forEach((s, i) => {
    readEffect(s, recipient, 'secondary', `secondaries.${i}.`, s.chance ?? null);
    readEffect(s.self, 'self', 'secondary', `secondaries.${i}.self.`, s.chance ?? null);
  });
  if (m.selfSwitch) effects.push({kind: 'self_switch', recipient: 'self', phase: 'primary',
    mode: m.selfSwitch, source_field: 'selfSwitch'});
  if (m.forceSwitch) effects.push({kind: 'force_switch', recipient: 'target', phase: 'primary',
    source_field: 'forceSwitch'});
  if (m.sideCondition) effects.push({kind: 'side_condition', recipient: m.target,
    phase: 'primary', condition: m.sideCondition, source_field: 'sideCondition'});
  moves[id] = {id, target: m.target, category: m.category, accuracy: m.accuracy,
    effects, callbacks: [...new Set(callbacks(m))].sort()};
}
process.stdout.write(JSON.stringify({schema: 'move-semantics-v1', format,
  showdown_version: require('pokemon-showdown/package.json').version, moves,
  scope: 'Potential effects from structured move fields. Callback code and battle-state activation conditions are not interpreted; absent effects are unknown, not impossible.'}, null, 2));
