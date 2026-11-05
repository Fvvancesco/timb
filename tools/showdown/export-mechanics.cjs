// Conservative static type model, exported from the project's pinned simulator.
const fs = require('node:fs');
const {Dex} = require('pokemon-showdown');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const dex = Dex.forFormat(input.format || 'gen9ou');
const types = dex.types.all().filter(t => t.exists && !t.isNonstandard && t.name !== 'Stellar').map(t => t.name).sort();
const chart = {};
for (const attack of types) {
  chart[attack] = {};
  for (const defense of types) {
    chart[attack][defense] = dex.getImmunity(attack, defense) ? 2 ** dex.getEffectiveness(attack, defense) : 0;
  }
}
const moves = {};
for (const id of [...new Set(input.move_ids || [])].sort()) {
  const m = dex.moves.get(id);
  const reasons = [];
  if (!m.exists) reasons.push('unknown_move');
  if (!['Physical', 'Special'].includes(m.category)) reasons.push('not_an_attack');
  if (m.damage || m.damageCallback || m.ohko) reasons.push('nonstandard_damage');
  if (m.onModifyType || m.onModifyMove || m.onEffectiveness || m.ignoreImmunity) reasons.push('contextual_type_or_immunity');
  if (!types.includes(m.type)) reasons.push('unsupported_type');
  if (!['normal', 'any', 'randomNormal', 'allAdjacent', 'allAdjacentFoes', 'adjacentFoe'].includes(m.target)) reasons.push('unsupported_target');
  moves[id] = {id: m.id, type: m.type, category: m.category, eligible: reasons.length === 0, excluded_reasons: reasons};
}
const natures = Object.fromEntries(dex.natures.all().map(n => [n.name, {plus: n.plus || null, minus: n.minus || null}]));
process.stdout.write(JSON.stringify({natures, format: input.format || 'gen9ou',
  showdown_version: require('pokemon-showdown/package.json').version, types, chart, moves,
  scope: 'Initial types only; no damage calculation, abilities, items, Terastallization or battle-state effects.'}));
