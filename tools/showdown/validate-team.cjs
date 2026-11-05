const fs = require('node:fs');
const {Teams, TeamValidator} = require('pokemon-showdown');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const errors = new TeamValidator(input.format).validateTeam(structuredClone(input.sets)) || [];
process.stdout.write(JSON.stringify({legal:errors.length === 0, errors,
  text:Teams.export(input.sets), packed:Teams.pack(input.sets)}));
