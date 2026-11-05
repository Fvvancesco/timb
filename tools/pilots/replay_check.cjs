// Replays saved simulator input, not the stochastic Foul Play policy.
const fs = require('fs');
const crypto = require('crypto');
const {BattleStream} = require(process.argv[2] + '/dist/sim/battle-stream.js');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const original = fs.readFileSync(process.argv[4], 'utf8');
const event = line => /^\|(-[^|]+|turn|move|switch|drag|faint|win|upkeep|cant|detailschange|replace|swap|transform)(\||$)/.test(line) || /^\|tie(?:\||$)/.test(line);
const select = text => text.split('\n').filter(event);
(async () => {
  const stream = new BattleStream({noCatch:true, replay:'spectator'});
  const chunks=[]; const reading=(async()=>{for await (const chunk of stream) chunks.push(chunk)})();
  await stream.write(data.inputLog.join('\n') + (data.forced_stop ? '\n>forcetie' : ''));
  if (!stream.battle.ended) await stream.writeEnd();
  await reading;
  const replay=chunks.join('\n');
  const observed=select(original), rerun=select(replay);
  const match=JSON.stringify(observed)===JSON.stringify(rerun);
  const out={match, seed:data.seed, original_events:observed.length, replay_events:rerun.length,
    replay_sha256:crypto.createHash('sha256').update(rerun.join('\n')).digest('hex'),
    first_difference: match ? null : {index:observed.findIndex((v,i)=>v!==rerun[i]), original: observed.find((v,i)=>v!==rerun[i]) ?? null, replay: rerun.find((v,i)=>v!==observed[i]) ?? null}};
  fs.writeFileSync(process.argv[5],JSON.stringify(out,null,2)+'\n');
  fs.writeFileSync(process.argv[5]+'.log',replay);
  console.log(JSON.stringify(out)); if (!match) process.exitCode=1;
})().catch(e=>{console.error(e.stack);process.exitCode=1});
