// Logic-only checks of the labelling page with a fake DOM. Does NOT open a browser or play video.
//   node tests/test_label_points_ui.cjs                      -> template + synthetic run data
//   node tests/test_label_points_ui.cjs <label.html> [out]   -> a built page; optionally write the export to <out>
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const SYNTHETIC = {
  page_version: 'label-points-1', labels_schema_version: 1, run_id: 'a'.repeat(64), source_video_sha256: 'b'.repeat(64),
  video_hash_basis: 'synthetic', fps: 30, source_start_frame: 1000, frames: 300, run_kind: 'replay', run_path: '/synthetic',
  video_src: '../replay/source.mp4', video_check: {status: 'aligned'},
  shot_types: ['serve', 'forehand', 'backhand', 'volley', 'overhead', 'other', 'unsure'], hitters: ['near', 'far'],
  bounce_calls: ['in', 'out', 'unsure'], point_sides: ['near', 'far', 'unknown'], coverage_kinds: ['shots', 'bounces', 'points'],
  candidates: [
    {id: 'hit-50', kind: 'hit', source_frame: 1050, run_frame: 50, player_id: 'near', shot_type: 'forehand', review_status: 'unreviewed'},
    {id: 'bounce-60', kind: 'bounce', source_frame: 1060, run_frame: 60, player_id: null, shot_type: null, review_status: 'unreviewed'},
  ],
  candidate_limitation: 'hints only',
};
const pagePath = process.argv[2];
let html = fs.readFileSync(pagePath || path.join(__dirname, '../tennis_vision/label_points.html'), 'utf8');
if (!pagePath) html = html.replace('__LABEL_DATA__', JSON.stringify(SYNTHETIC));
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const D = JSON.parse(scripts.find(s => s[1].includes('application/json'))[2]);
const code = scripts.find(s => !s[1].includes('application/json'))[2];
assert.ok(!/fetch\(|XMLHttpRequest|WebSocket|sendBeacon/.test(code), 'page must not send anything anywhere');

class Element {
  constructor(id) {
    Object.assign(this, {id, children: [], handlers: {}, textContent: '', hidden: false, checked: true, width: 1200, height: 64,
      currentTime: 0, duration: NaN, paused: true, readyState: 1, className: '', focused: false});
    this._value = '';
  }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }
  append(...v) { this.children.push(...v); }
  replaceChildren(...v) { this.children = v; }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  setAttribute(name, value) { this[name] = String(value); }
  click() { return this.onclick?.(); }
  focus() { this.focused = true; }
  blur() { this.focused = false; }
  getBoundingClientRect() { return {left: 0, width: 1200}; }
  getContext() { const g = {}; for (const f of ['clearRect', 'fillRect', 'beginPath', 'stroke']) g[f] = () => {}; return g; }
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
  load() {}
}
const elements = new Map();
const created = [];
// Elements the page creates and gives an id are found by id, like in a browser (latest wins).
const get = id => {
  const made = created.findLast(e => e.id === id);
  if (made) return made;
  if (!elements.has(id)) elements.set(id, new Element(id));
  return elements.get(id);
};
get('label-data').textContent = JSON.stringify(D);
get('speed').value = '0.5';
for (const id of ['shortcutHelp', 'prompt', 'hint', 'videoProblem', 'editor']) get(id).hidden = true; // hidden in the HTML
const saved = new Map();
let download = null, downloadName = null;
const docHandlers = {};
const context = vm.createContext({
  document: {getElementById: get, createElement: tag => { const e = new Element(''); e.tagName = tag; created.push(e); return e; },
    addEventListener: (type, handler) => { docHandlers[type] = handler; }},
  localStorage: {getItem: k => saved.get(k) ?? null, setItem: (k, v) => saved.set(k, v)},
  Blob, URL: {createObjectURL: b => { download = b; return 'blob:fake'; }, revokeObjectURL() {}},
  setTimeout: f => f(), Date, console,
});
vm.runInContext(code, context);
const run = s => vm.runInContext(s, context);
const video = get('video');
const START = D.source_start_frame, N = D.frames;
assert.ok(N >= 200, 'fixture needs at least 200 frames');
function press(key, opts = {}) {
  const e = {key, code: key === ' ' ? 'Space' : '', shiftKey: false, ctrlKey: false, metaKey: false, altKey: false, repeat: false,
    target: {tagName: 'BODY'}, defaultPrevented: false, ...opts};
  e.preventDefault = () => { e.defaultPrevented = true; };
  docHandlers.keydown(e);
  return e;
}
const src = () => run('srcNow()');
const L = () => JSON.parse(run('JSON.stringify(L)'));
const notice = () => get('notice').textContent;
const goto = sf => run(`seekSource(${sf})`);

// ---- start-up: video link, candidates shown as hints, nothing labelled
assert.equal(video.src, D.video_src);
assert.equal(src(), START);
assert.equal(get('candidateList').children.length, D.candidates.length);
assert.ok(get('candidateList').children.every(c => / hint/.test(c.className)), 'candidates are styled as hints');
assert.deepEqual([L().shots.length, L().bounces.length, L().points.length, L().coverage.length], [0, 0, 0, 0]);
assert.match(get('readout').textContent, new RegExp(`source frame ${START} · run frame 0`));

// ---- frame stepping and play/pause
press('ArrowRight'); assert.equal(src(), START + 1);
press('ArrowRight', {repeat: true}); assert.equal(src(), START + 2, 'held arrows repeat frame steps');
press('ArrowLeft'); assert.equal(src(), START + 1);
press('ArrowRight', {shiftKey: true}); assert.equal(src(), START + 1 + Math.round(D.fps));
press('ArrowLeft', {shiftKey: true}); press('ArrowLeft', {shiftKey: true}); assert.equal(src(), START, 'clamped at the first frame');
press(' '); assert.equal(video.paused, false); press(' '); assert.equal(video.paused, true);
get('nextFrame').click(); assert.equal(src(), START + 1); get('previousFrame').click(); assert.equal(src(), START);
// Frame <-> time: seeking puts the playhead inside the frame, not on its boundary.
goto(START + 123); assert.equal(src(), START + 123); assert.equal(run('runFrame()'), 123);
assert.equal(get('videoProblem').hidden, D.video_check.status !== 'missing', 'a missing video asks for the file');
get('videoProblem').hidden = true;
video.handlers.seeked(); assert.equal(get('videoProblem').hidden, true, 'a seek that lands is fine');
// A seek that does not land (e.g. a server without range requests) is reported, and the readout shows the real frame.
goto(START + 40); video.currentTime = 0; video.handlers.seeked();
assert.equal(get('videoProblem').hidden, false); assert.match(get('videoProblemText').textContent, /did not move/);
assert.equal(src(), START); get('videoProblem').hidden = true;
goto(START);

// ---- machine candidates are hints only
if (D.candidates.length) {
  const first = [...D.candidates].sort((a, b) => a.source_frame - b.source_frame).find(c => c.source_frame > START);
  press('.'); assert.equal(src(), first.source_frame); assert.match(notice(), /hint only/);
  assert.equal(get('hint').hidden, false); assert.match(get('hint').textContent, /not a label/);
  assert.equal(L().shots.length + L().bounces.length, 0, 'jumping to a candidate labels nothing');
  press(','); assert.ok(src() < first.source_frame || /No earlier/.test(notice()));
  goto(START);
}
press('.', {repeat: true}); // repeats of non-step keys are ignored
goto(START + N - 1); press('.'); assert.match(notice(), /No later machine candidate|no machine candidates/);

// ---- shot: H, hitter, type
const shotFrame = START + 50;
goto(shotFrame);
press('h'); assert.equal(get('prompt').hidden, false); assert.match(get('prompt').textContent, new RegExp(`source frame ${shotFrame}`));
press('x'); assert.match(notice(), /not an answer/); assert.equal(L().shots.length, 0);
press('n'); assert.match(get('prompt').textContent, /type\?/);
press('2');
assert.equal(get('prompt').hidden, true);
assert.deepEqual(L().shots, [{id: 'shot-1', source_frame: shotFrame, hitter: 'near', shot_type: 'forehand', decision: 'human'}]);
press('h'); assert.match(notice(), /already labelled/); assert.equal(get('prompt').hidden, true);
// Escape cancels without labelling; repeats and keys typed into inputs never label.
goto(shotFrame + 5); press('h'); press('Escape'); assert.equal(get('prompt').hidden, true); assert.match(notice(), /nothing was labelled/);
press('h', {repeat: true}); assert.equal(get('prompt').hidden, true);
for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT']) { const e = press('h', {target: {tagName}}); assert.equal(e.defaultPrevented, false); }
assert.equal(get('prompt').hidden, true);
press('h', {metaKey: true}); assert.equal(get('prompt').hidden, true);
// Far hitter, every type key maps to the decided list.
const types = {'1': 'serve', '2': 'forehand', '3': 'backhand', '4': 'volley', '5': 'overhead', '6': 'other', '7': 'unsure'};
let f = shotFrame + 10;
for (const [k, type] of Object.entries(types)) { goto(f); press('H'); press('F'); press(k); assert.equal(L().shots.at(-1).shot_type, type); f += 3; }
assert.equal(L().shots.length, 8);
assert.ok(L().shots.slice(1).every(s => s.hitter === 'far'));

// ---- undo / delete
press('z'); assert.equal(L().shots.length, 7);
press('z', {ctrlKey: true}); assert.equal(L().shots.length, 6);
run(`select('shots','shot-1')`); press('Delete'); assert.equal(L().shots.some(s => s.id === 'shot-1'), false);
press('z'); assert.equal(L().shots.some(s => s.id === 'shot-1'), true, 'undo restores a removed label');
// Editing a selected label through the editor (select) is undoable.
run(`select('shots','shot-1')`); assert.equal(get('editor').hidden, false);
get('edit-shot_type').value = 'backhand'; get('edit-shot_type').onchange();
assert.equal(L().shots.find(s => s.id === 'shot-1').shot_type, 'backhand');
press('z'); assert.equal(L().shots.find(s => s.id === 'shot-1').shot_type, 'forehand');

// ---- bounce
goto(START + 60); press('b'); press('i');
goto(START + 61); press('B'); press('O');
goto(START + 62); press('b'); press('u');
assert.deepEqual(L().bounces.map(b => [b.source_frame, b.call]), [[START + 60, 'in'], [START + 61, 'out'], [START + 62, 'unsure']]);
goto(START + 60); press('b'); assert.match(notice(), /already labelled/);

// ---- points: start/server, end/winner, score text
press(']'); assert.match(notice(), /No open point/);
goto(START + 5); press('['); press('n');
assert.deepEqual(L().open.point, {source_start_frame: START + 5, server: 'near'});
press('['); assert.match(notice(), /already open/);
goto(START + 3); press(']'); assert.match(notice(), /before the point's start/);
// Export is refused while a point is open.
get('labellerName').value = 'Synthetic tester';
get('export').click(); assert.equal(download, null); assert.match(notice(), /still open/);
goto(START + 100); press(']'); press('f');
assert.equal(L().points.length, 1);
assert.deepEqual(L().points[0], {id: L().points[0].id, source_start_frame: START + 5, source_end_frame: START + 100,
  server: 'near', winner: 'far', decision: 'human'});
assert.equal(L().open.point, null);
goto(START + 50); press('['); assert.match(notice(), /inside point/);
// Overlap on the way out: a point started before an existing one cannot end inside it.
goto(START + 101); press('['); press('u'); goto(START + 150); press(']'); press('u');
assert.equal(L().points[1].server, 'unknown'); assert.equal(L().points[1].winner, 'unknown');
press('t'); assert.equal(get('edit-score_text').focused, true);
get('edit-score_text').value = '15-0'; get('edit-score_text').onchange();
assert.equal(L().points[1].score_text, '15-0');
press('z'); assert.equal(L().points[1].score_text, undefined); press('t');
get('edit-score_text').value = '15-0'; get('edit-score_text').onkeydown({key: 'Enter'});
assert.equal(L().points[1].score_text, '15-0');

// ---- coverage: explicit start and end, never assumed
assert.equal(L().coverage.length, 0, 'no coverage until John marks it');
goto(START + 120); press('c'); assert.equal(L().open.coverage.source_start_frame, START + 120);
goto(START + 110); press('c'); assert.match(notice(), /before its start/);
get('covShots').checked = false; get('covBounces').checked = false; get('covPoints').checked = false;
goto(START + 160); press('c'); assert.match(notice(), /Tick at least one/);
get('covShots').checked = true; get('covBounces').checked = true;
press('c');
assert.deepEqual(L().coverage.map(c => [c.source_start_frame, c.source_end_frame, c.kinds]), [[START + 120, START + 160, ['shots', 'bounces']]]);
goto(START + 150); press('c'); goto(START + 170); get('covPoints').checked = false; press('c'); assert.match(notice(), /overlap/);
press('z'); // undo the second span's start
assert.equal(L().open.coverage, null);
goto(START); press('c'); goto(START + 119); get('covPoints').checked = true; press('c');
assert.equal(L().coverage.length, 2);

// ---- export
get('labellerName').value = '';
get('export').click(); assert.equal(download, null); assert.match(notice(), /labeller name/);
get('labellerName').value = 'Synthetic tester';
get('fileNotes').value = 'synthetic fake-DOM export';
get('export').click();
assert.ok(download, 'export produced a file');
assert.ok(created.some(e => e.download && /^labels-[0-9a-f]{8}-\d{4}-\d{2}-\d{2}\.json$/.test(e.download)));
(async () => {
  const text = await download.text();
  const payload = JSON.parse(text);
  assert.equal(payload.schema_version, 1);
  assert.equal(payload.kind, 'tennis_ground_truth_labels');
  assert.deepEqual(payload.binding, {run_id: D.run_id, source_video_sha256: D.source_video_sha256, fps: D.fps});
  assert.deepEqual(payload.shot_types, D.shot_types);
  assert.equal(payload.labeller.name, 'Synthetic tester'); assert.match(payload.labeller.date, /^\d{4}-\d{2}-\d{2}$/);
  assert.equal(payload.notes, 'synthetic fake-DOM export');
  assert.equal(payload.shots.length, 6); assert.equal(payload.bounces.length, 3); assert.equal(payload.points.length, 2);
  assert.ok(payload.shots.every(s => s.decision === 'human' && s.source_frame >= START && s.source_frame < START + N));
  assert.ok(payload.coverage.every(c => !('id' in c)), 'internal span ids are not exported');
  assert.ok(payload.points.every(p => Object.keys(p).every(k => ['id', 'source_start_frame', 'source_end_frame', 'server', 'winner', 'score_text', 'decision', 'notes'].includes(k))));
  const ids = [...payload.shots, ...payload.bounces, ...payload.points].map(e => e.id);
  assert.equal(new Set(ids).size, ids.length, 'ids unique across lists');
  assert.ok(!text.includes('candidate'), 'no machine candidate is exported');
  // checkPayload mirrors the Python validator's main rules.
  context.fixture = JSON.parse(text);
  run('checkPayload(fixture)');
  context.fixture.binding.run_id = 'c'.repeat(64); assert.throws(() => run('checkPayload(fixture)'), /another run/);
  context.fixture = JSON.parse(text); context.fixture.shots.push({...context.fixture.shots[0], id: 'dup-frame'});
  assert.throws(() => run('checkPayload(fixture)'), /Two shots/);
  // Import an export into a fresh state; it round-trips and is undoable.
  run('commit(()=>{L=blank();})'); assert.equal(L().shots.length, 0);
  context.fixture = JSON.parse(text); run('importLabels(fixture)');
  assert.equal(L().shots.length, 6); assert.equal(L().points[1].score_text, '15-0');
  goto(START + 200); press('h'); press('n'); press('7');
  assert.ok(!payload.shots.some(s => s.id === L().shots.at(-1).id), 'new ids after import do not collide');
  press('z');
  context.fixture = JSON.parse(text); context.fixture.binding.source_video_sha256 = 'd'.repeat(64);
  assert.throws(() => run('importLabels(fixture)'), /another run or video/);
  // Draft autosave is per run, is not an export, and a blocked storage never blocks labelling or export.
  assert.ok(saved.has('tennis-labels-draft:' + D.run_id));
  assert.match(get('draftState').textContent, /not an export/);
  context.localStorage.setItem = () => { throw Error('blocked'); };
  goto(START + 210); press('b'); press('u'); assert.match(get('draftState').textContent, /unavailable/);
  press('z');
  download = null; get('export').click(); assert.ok(download);
  // Help toggles.
  press('?'); assert.equal(get('shortcutHelp').hidden, false); press('Escape'); assert.equal(get('shortcutHelp').hidden, true);
  if (process.argv[3]) fs.writeFileSync(process.argv[3], text);
  console.log('Label page UI logic checks passed (fake DOM; not a visual/browser playback test).');
})().catch(error => { console.error(error); process.exitCode = 1; });
