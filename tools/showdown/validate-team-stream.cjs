// Sequential JSONL validator: load the pinned Dex once, never cache team verdicts.
const readline = require('node:readline');
const {TeamValidator} = require('pokemon-showdown');
process.stdout.write(JSON.stringify({ready: true,
  showdown_version: require('pokemon-showdown/package.json').version}) + '\n');
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const input = JSON.parse(line);
    const errors = new TeamValidator(input.format).validateTeam(structuredClone(input.sets)) || [];
    process.stdout.write(JSON.stringify({legal: errors.length === 0, errors}) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({error: String(error)}) + '\n');
  }
});
