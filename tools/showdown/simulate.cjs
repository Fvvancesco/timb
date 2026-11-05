// Integration pilot only: max listed base power, no voluntary switches or Tera.
// Each policy sees only its player stream. Full teams are available to the engine.
const fs = require('node:fs');
const crypto = require('node:crypto');
const {performance} = require('node:perf_hooks');
const {BattleStream, getPlayerStreams, Teams, TeamValidator, Dex} = require('pokemon-showdown');
const {RandomPlayerAI} = require('pokemon-showdown/dist/sim/tools/random-player-ai');

function seedFor(root, label) {
  const bytes = crypto.createHash('sha256').update(`icon-v1:${root}:${label}`).digest();
  return [0, 2, 4, 6].map(i => bytes.readUInt16BE(i));
}

async function main(input) {
  const started = performance.now();
  if (input.format !== 'gen9ou') throw new Error('This integration pilot supports gen9ou only');
  if (!Number.isInteger(input.seed)) throw new Error('An integer seed is required');
  const maxTurns = input.max_turns ?? 200;
  if (!Number.isInteger(maxTurns) || maxTurns < 1) throw new Error('Invalid turn limit');
  const validator = new TeamValidator(input.format);
  for (const team of [input.team_a, input.team_b]) {
    if (!Array.isArray(team) || team.length !== 6) throw new Error('Six members required');
    const errors = validator.validateTeam(structuredClone(team));
    if (errors) throw new Error(errors.join('; '));
  }
  const dex = Dex.forFormat(input.format);
  class MaxBasePowerAI extends RandomPlayerAI {
    chooseMove(active, moves) {
      const power = m => dex.moves.get(m.move.move).basePower;
      const maximum = Math.max(...moves.map(power));
      return this.prng.sample(moves.filter(m => power(m) === maximum)).choice;
    }
  }
  class LimitedStream extends BattleStream {
    truncated = false;
    _write(chunk) {
      super._write(chunk);
      // Stop before choices for turn maxTurns+1 can execute.
      if (this.battle && !this.battle.ended && this.battle.turn > maxTurns) {
        this.truncated = true;
        super._write('>forcetie');
      }
    }
  }
  const engine = new LimitedStream({noCatch:true});
  const streams = getPlayerStreams(engine);
  const seeds = {engine:seedFor(input.seed, 'engine'),
    p1:seedFor(input.seed, 'p1'), p2:seedFor(input.seed, 'p2')};
  const players = [new MaxBasePowerAI(streams.p1, {seed:seeds.p1}),
    new MaxBasePowerAI(streams.p2, {seed:seeds.p2})];
  const chunks = [];
  const jobs = [players[0].start(), players[1].start(), (async () => {
    for await (const chunk of streams.omniscient) chunks.push(chunk);
  })()];
  const finished = Promise.all(jobs);
  await streams.omniscient.write(`>start ${JSON.stringify({formatid:input.format, seed:seeds.engine})}
>player p1 ${JSON.stringify({name:'P1', team:Teams.pack(input.team_a)})}
>player p2 ${JSON.stringify({name:'P2', team:Teams.pack(input.team_b)})}`);
  await finished;
  const log = chunks.join('\n');
  // Showdown adds wall-clock timestamps; they are not part of gameplay determinism.
  const gameplayLog = log.split('\n').filter(line => !line.startsWith('|t:|')).join('\n');
  const winner = log.match(/^\|win\|(P[12])$/m)?.[1] ?? null;
  const status = engine.truncated ? 'truncated' : winner ? 'completed' :
    /^\|tie$/m.test(log) ? 'draw' : 'error';
  return {format:input.format, showdown_version:require('pokemon-showdown/package.json').version,
    pilot:'max-listed-base-power-v1', root_seed:input.seed, seeds, max_turns:maxTurns,
    status, outcome:status === 'completed' ? (winner === 'P1' ? 1 : 0) : null,
    winner:status === 'completed' ? winner : null,
    completed_turns:engine.truncated ? maxTurns : engine.battle.turn,
    elapsed_seconds:(performance.now() - started) / 1000,
    team_a:input.team_a, team_b:input.team_b, log,
    gameplay_sha256:crypto.createHash('sha256').update(gameplayLog).digest('hex')};
}

main(JSON.parse(fs.readFileSync(0, 'utf8'))).then(result => {
  process.stdout.write(JSON.stringify(result));
}).catch(error => {console.error(error.stack); process.exitCode = 1;});
