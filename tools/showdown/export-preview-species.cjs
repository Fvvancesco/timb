// Resolve the public Team Preview names in the pinned species database.
const fs = require('node:fs');
const {Dex} = require('pokemon-showdown');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const dex = Dex.forFormat('gen9ou');
const species = {};
for (const name of input.names) {
  const entry = dex.species.get(name);
  if (entry.exists && !name.endsWith('-*')) {
    species[name] = {id: entry.id, base_species: entry.baseSpecies, types: entry.types, stats: entry.baseStats};
  }
}
process.stdout.write(JSON.stringify(species));
