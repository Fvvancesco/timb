"""Create a narrowly patched, loopback-only Showdown 0.11.11 worker copy."""
import hashlib, json, pathlib, shutil, sys
source=pathlib.Path(sys.argv[1]).resolve(); dest=pathlib.Path(sys.argv[2]).resolve(); port=int(sys.argv[3]) if len(sys.argv)>3 else 18766
assert 18766 <= port <= 18779
assert json.loads((source/'package.json').read_text())['version']=='0.11.11'
if dest.exists(): raise SystemExit('Refusing to overwrite existing server copy')
shutil.copytree(source,dest,symlinks=True)
(dest/'logs/repl').mkdir(parents=True,exist_ok=True)
(dest/'config/chat-plugins').mkdir(parents=True,exist_ok=True)
(dest/'config/config.js').write_text(f"exports.bindaddress = '127.0.0.1';\nexports.port = {port};\nexports.serverid = 'localhost';\nexports.noguestsecurity = true;\nexports.nothrottle = true;\n")
def replace_once(file,old,new):
    text=file.read_text()
    if text.count(old)!=1: raise RuntimeError(f'Unexpected source: {file}: expected exactly one match')
    file.write_text(text.replace(old,new,1))
ladder=dest/'dist/server/ladders.js'
replace_once(ladder,'    return Rooms.createBattle({\n      format: formatid,\n      players,', '''    const iconSpec = JSON.parse(require("fs").readFileSync(process.env.ICON_JOB_SPEC, "utf8"));
    if (Config.bindaddress !== "127.0.0.1" || !Config.nothrottle || formatid !== iconSpec.format) throw new Error("Invalid ICON server configuration");
    if (players.length !== 2 || new Set(players.map(p => p.user.id)).size !== 2) throw new Error("Invalid ICON player count");
    const desired = iconSpec.players.map(x => x.toLowerCase().replace(/[^a-z0-9]/g, ""));
    players.sort((a,b) => desired.indexOf(a.user.id)-desired.indexOf(b.user.id));
    if (players.some((p,i) => p.user.id !== desired[i])) throw new Error("Unexpected ICON players");
    return Rooms.createBattle({
      seed: iconSpec.engine_seed,
      format: formatid,
      players,''')
battle=dest/'dist/server/room-battle.js'
replace_once(battle,'    this.inputLog = null;\n    this.turn = 0;', '''    this.inputLog = null;
    this.iconForced = null;
    const iconSpec = JSON.parse(require("fs").readFileSync(process.env.ICON_JOB_SPEC, "utf8"));
    this.iconMaxTurns = iconSpec.max_turns;
    if (!Number.isInteger(this.iconMaxTurns) || this.iconMaxTurns < 1 || this.iconMaxTurns > 200 || iconSpec.wall_limit_seconds !== 120) throw new Error("Invalid ICON bounds");
    this.iconWallTimer = setTimeout(() => { if (!this.ended && !this.iconForced) { this.iconForced = "wall_timeout"; void this.stream.write(">forcetie"); } }, iconSpec.wall_limit_seconds * 1000);
    this.turn = 0;''')
replace_once(battle,'            this.turn = parseInt(line.slice(6));', '''            this.turn = parseInt(line.slice(6));
            if (this.turn > this.iconMaxTurns && !this.iconForced) { this.iconForced = "turn_limit"; void this.stream.write(">forcetie"); }''')
replace_once(battle,'        this.inputLog = this.logData.inputLog;\n        this.started = true;', '''        this.inputLog = this.logData.inputLog;
        clearTimeout(this.iconWallTimer);
        const iconFs = require("fs"), iconPath = require("path");
        const iconOutput = iconPath.join(process.env.ICON_OUTPUT_DIR, this.roomid + ".json");
        const iconData = {forced_stop: this.iconForced, current_request_turn: this.turn, ...this.logData};
        iconFs.writeFileSync(iconOutput + ".tmp", JSON.stringify(iconData));
        iconFs.renameSync(iconOutput + ".tmp", iconOutput);
        this.started = true;''')
files=[source/'dist/server/ladders.js',source/'dist/server/room-battle.js',ladder,battle,dest/'config/config.js']
(dest/'icon-patch-hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(dest)
