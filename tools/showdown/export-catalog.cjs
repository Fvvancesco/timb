// The only authoritative legality check in this prototype is Showdown's validator.
const fs = require('node:fs');
const crypto = require('node:crypto');
const {Teams, Dex, TeamValidator, toID} = require('pokemon-showdown');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const format = input.format || 'gen9ou';
const dex = Dex.forFormat(format);
const validator = new TeamValidator(format);
const catalog = new Map();
const teams = [];
const rejectedSets = [];
for (const source of input.teams) {
  const imported = Teams.import(source.text);
  if (!imported) throw new Error(`Cannot parse ${source.path}`);
  const teamErrors = validator.validateTeam(structuredClone(imported)) || [];
  const ids = [];
  for (const original of imported) {
    const normalized = structuredClone(original);
    const errors = validator.validateTeam([normalized]) || [];
    if (errors.length) {
      rejectedSets.push({source:source.path, species:original.species, errors});
      continue;
    }
    normalized.name = normalized.species;
    normalized.moves.sort();
    const packed = Teams.pack([normalized]);
    const id = crypto.createHash('sha256').update(packed).digest('hex').slice(0, 20);
    ids.push(id);
    if (!catalog.has(id)) {
      const species = dex.species.get(normalized.species);
      catalog.set(id, {
        id, species: species.name, species_id: species.id,
        base_species: species.baseSpecies, types: species.types,
        base_stats: species.baseStats, set: normalized, packed,
        ability_id:toID(normalized.ability), item_id:toID(normalized.item),
        source_sets: [],
        provenance: [],
        moves: normalized.moves.map(name => {
          const m = dex.moves.get(name);
          return {id:m.id, name:m.name, type:m.type, category:m.category,
            base_power:m.basePower, priority:m.priority, target:m.target,
            heal:m.heal || null, side_condition:m.sideCondition || null};
        }),
      });
    }
    catalog.get(id).provenance.push(source.path);
    catalog.get(id).source_sets.push({source:source.path, parsed:original});
  }
  teams.push({source:source.path, legal:teamErrors.length === 0,
    validation_errors:teamErrors, set_ids:ids,
    species_composition:imported.map(s=>dex.species.get(s.species).baseSpecies).sort()});
}
process.stdout.write(JSON.stringify({format,
  showdown_version:require('pokemon-showdown/package.json').version,
  sets:[...catalog.values()].sort((a,b)=>a.id.localeCompare(b.id)), teams,
  rejected_sets:rejectedSets}, null, 2));
