// Logic-only unit test with a fake DOM for the match dashboard. Does NOT launch a browser.
//   node tests/test_match_dashboard_ui.cjs                    -> template + the hand-written fixture below
//   node tests/test_match_dashboard_ui.cjs <dashboard.html>   -> generic checks on a built page
//   node tests/test_match_dashboard_ui.cjs <html> --synthetic -> plus checks for tests/test_match_dashboard.py's table
//   node tests/test_match_dashboard_ui.cjs --print-fixture    -> print the fixture (Python checks its columns)
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Same column order as tennis_vision/shot_table.py (checked by tests/test_match_dashboard.py).
const SHOT_FIELDS = [['contact', ['contact_frame', 'contact_source_frame', 'contact_time_s']], ['hitter', ['hitter']],
  ['shot_status', ['shot_status']], ['shot_type', ['shot_type']], ['hitter_position', ['hitter_position_x_m', 'hitter_position_y_m']],
  ['opponent_position', ['opponent_position_x_m', 'opponent_position_y_m']], ['landing', ['landing_x_m', 'landing_y_m']],
  ['landing_call', ['landing_call']], ['direction', ['direction']], ['depth', ['depth']], ['ball_speed', ['ball_speed_kmh']],
  ['point', ['point_id']], ['rally_shot_index', ['rally_shot_index']]];
const POINT_FIELDS = [['start', ['start_frame', 'start_source_frame', 'start_time_s']], ['end', ['end_frame', 'end_source_frame', 'end_time_s']],
  ['server', ['server']], ['shot_count', ['shot_count']], ['winner', ['winner']], ['score_text', ['score_text']]];

function row(keys, fields, values) {
  const out = {};
  keys.forEach(k => { out[k] = values[k] ?? null; });
  for (const [name, cols] of fields) {
    const source = values[name + '_source'] || 'unknown';
    cols.forEach(c => { out[c] = source === 'unknown' ? null : (values[c] ?? null); });
    out[name + '_source'] = source;
    out[name + '_basis'] = values[name + '_basis'] || (source === 'unknown' ? 'no supported value' : 'fixture ' + source);
  }
  return out;
}
function shot(id, frame, v) {
  return row(['shot_id', 'candidate_event_id'], SHOT_FIELDS, Object.assign({shot_id: id, candidate_event_id: id, contact_frame: frame,
    contact_source_frame: frame + 1000, contact_time_s: frame / 30, contact_source: 'estimated'}, v));
}
const P = (x, y, s = 'observed') => ({hitter_position_x_m: x, hitter_position_y_m: y, hitter_position_source: s});
const L = (x, y, s) => ({landing_x_m: x, landing_y_m: y, landing_source: s});
const HC = 'human_confirmed', EST = 'estimated';
const fixture = {
  schema_version: 1, kind: 'match_dashboard',
  table: {folder: 'fixture-table', sha256: 'f'.repeat(64), schema_version: 1},
  replay: {folder: 'fixture-replay', review_run_id: 'run-fixture-0001', fps: 30, frames: 300, source_start_frame: 1000},
  labels: {file: 'labels.json', labeller: 'test', labelled_on: '2026-10-08'},
  calibration: {status: 'reviewed', reviewed: true},
  limitations: ['Coverage and source counts are not accuracy.'],
  replay_link: {href: 'replay.html', status: 'fixture', seek: 'replay.html does not seek from a link yet.'},
  shots: [
    shot('a', 10, Object.assign({hitter: 'near', hitter_source: HC, shot_status: 'labelled_shot', shot_status_source: HC,
      shot_type: 'serve', shot_type_source: HC, direction: 'cross_court', direction_source: EST, depth: 'mid', depth_source: EST,
      ball_speed_kmh: 72.5, ball_speed_source: EST, point_id: 'p1', point_source: HC, rally_shot_index: 1, rally_shot_index_source: HC,
      landing_basis: '<img src=x onerror=alert(1)>'}, P(3, 0.5), L(7, 16, 'observed'))),
    shot('b', 40, Object.assign({hitter: 'far', hitter_source: HC, shot_status: 'labelled_shot', shot_status_source: HC,
      shot_type: 'forehand', shot_type_source: HC, depth: 'deep', depth_source: EST, point_id: 'p1', point_source: HC,
      rally_shot_index: 2, rally_shot_index_source: HC}, P(8, 23.5), L(2, 3, HC))),
    shot('c', 70, Object.assign({hitter: 'near', hitter_source: HC, shot_status: 'labelled_shot', shot_status_source: HC,
      shot_type: 'backhand', shot_type_source: HC, landing_call: 'out', landing_call_source: HC, depth: 'deep', depth_source: EST,
      point_id: 'p1', point_source: HC, rally_shot_index: 3, rally_shot_index_source: HC}, P(3, 0.5), L(5, 20, EST))),
    shot('d', 100, Object.assign({hitter: 'far', hitter_source: HC, shot_status: 'labelled_shot', shot_status_source: HC,
      shot_type: 'forehand', shot_type_source: HC, point_id: 'p1', point_source: HC, rally_shot_index: 4, rally_shot_index_source: HC}, P(8, 23.5))),
    shot('e', 160, Object.assign({hitter: 'far', hitter_source: EST, shot_status: 'shot_candidate', shot_status_source: EST,
      shot_type: 'serve', shot_type_source: EST, direction: 'cross_court', direction_source: EST, depth: 'short', depth_source: EST,
      point_id: 'p2', point_source: EST, rally_shot_index: 1, rally_shot_index_source: EST}, P(8, 23.5), L(4, 8, 'observed'))),
    shot('f', 200, Object.assign({hitter: 'near', hitter_source: EST, shot_status: 'uncertain_contact', shot_status_source: EST,
      point_id: 'p2', point_source: EST}, P(3, 0.5))),
    shot('g', 280, Object.assign({hitter: 'near', hitter_source: EST, shot_status: 'shot_candidate', shot_status_source: EST,
      shot_type: 'forehand', shot_type_source: EST}, P(40, 0.5))),
  ],
  points: [
    row(['point_id'], POINT_FIELDS, {point_id: 'p1', start_frame: 5, start_source_frame: 1005, start_time_s: 5 / 30, start_source: HC,
      end_frame: 120, end_source_frame: 1120, end_time_s: 4, end_source: HC, server: 'near', server_source: HC,
      shot_count: 4, shot_count_source: HC, winner: 'near', winner_source: HC, score_text: '15-0', score_text_source: HC}),
    row(['point_id'], POINT_FIELDS, {point_id: 'p2', start_frame: 150, start_source_frame: 1150, start_time_s: 5, start_source: HC,
      end_frame: 260, end_source_frame: 1260, end_time_s: 260 / 30, end_source: HC, server: 'far', server_source: HC,
      shot_count: 1, shot_count_source: EST}),
  ],
};

const args = process.argv.slice(2);
if (args.includes('--print-fixture')) { process.stdout.write(JSON.stringify(fixture)); process.exit(0); }
const pagePath = args.find(a => !a.startsWith('--'));
const synthetic = args.includes('--synthetic');
let html = fs.readFileSync(pagePath || path.join(__dirname, '../tennis_vision/match_dashboard.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const dataText = pagePath ? scripts.find(s => s[1].includes('application/json'))[2] : JSON.stringify(fixture).replace(/</g, '\\u003c');
const code = scripts.find(s => !s[1].includes('application/json'))[2];
const data = JSON.parse(dataText);

// Offline: no remote resources anywhere in the page.
assert.doesNotMatch(html, /(src|href)\s*=\s*["']?(https?:)?\/\//, 'no remote scripts, styles or fonts');
assert.ok(!code.includes('innerHTML'), 'table strings are inserted as text, never as HTML');

// ------------------------------------------------------------------ fake DOM
class Node {
  constructor(tag, ns) { this.tagName = tag; this.ns = ns; this.children = []; this.attrs = {}; this.handlers = {}; this._text = ''; this.hidden = false; this.checked = false; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this._text = String(v); this.children = []; }
  appendChild(c) { this.children.push(c); return c; }
  replaceChildren(...n) { this.children = n; this._text = ''; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, f) { (this.handlers[t] ||= []).push(f); }
  fire(t, ev = {}) { (this.handlers[t] || []).forEach(f => f(Object.assign({preventDefault() {}}, ev))); }
}
const elements = new Map();
const get = id => { if (!elements.has(id)) elements.set(id, new Node('div')); return elements.get(id); };
get('dashboard-data').textContent = dataText;
get('showEst').checked = true;
const root = new Node('html');
const store = new Map();
const storage = {getItem: k => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v))};
const context = vm.createContext({document: {getElementById: get, createElement: t => new Node(t), createElementNS: (ns, t) => new Node(t, ns),
  createTextNode: t => { const n = new Node('#text'); n.textContent = t; return n; }, documentElement: root}, localStorage: storage, console});
vm.runInContext(code, context);
// Results cross the vm realm boundary as JSON so deepEqual compares plain values.
const evaluate = expr => JSON.parse(vm.runInContext(`JSON.stringify(${expr})`, context));
const run = code => vm.runInContext(code, context);
const all = (node, pred, out = []) => { if (pred(node)) out.push(node); node.children.forEach(c => all(c, pred, out)); return out; };
const marks = () => all(get('court'), n => n.getAttribute('data-source') != null);
const chip = (group, value) => get(group).children.find(b => b.getAttribute('data-v') === value);
const setEst = on => { get('showEst').checked = on; get('showEst').fire('change'); };
const links = node => all(node, n => n.tagName === 'a');
const pointRowsDom = () => get('pointsBody').children;

// ------------------------------------------------------------------ invariants for any table
function checkInvariants(label) {
  const state = run('JSON.stringify(state)');
  const m = evaluate('MD.courtMarks(DATA, state)');
  const plotted = m.counts.human_confirmed + m.counts.observed + m.counts.estimated;
  assert.equal(marks().length, plotted, `${label}: one mark per plotted landing (${state})`);
  for (const node of marks()) assert.notEqual(node.getAttribute('data-source'), 'unknown', 'unknown landings are never plotted');
  if (!JSON.parse(state).showEst) assert.ok(marks().every(n => n.getAttribute('data-source') !== 'estimated'), 'estimated hidden');
  assert.match(get('courtCount').textContent, new RegExp(`${plotted} plotted · ${m.counts.unknown} unknown`));
  // Only shots, never contacts that the table does not call shots.
  for (const r of evaluate('MD.shotsInView(DATA, state)')) assert.ok(['labelled_shot', 'confirmed_shot', 'shot_candidate'].includes(r.shot_status));
  // Winners and servers come only from labels.
  for (const p of evaluate('MD.pointRows(DATA, state)')) {
    const src = data.points.find(q => q.point_id === p.id);
    if (p.winner) assert.equal(src.winner_source, 'human_confirmed');
    if (p.server) assert.equal(src.server_source, 'human_confirmed');
  }
  // Replay links only when the builder found the replay.
  if (!data.replay_link || !data.replay_link.href) assert.equal(links(get('shotsBody')).length, 0);
  const heat = evaluate('MD.heatGrid(DATA, state)');
  const cells = all(get('heat'), n => n.getAttribute('data-count') != null);
  assert.equal(cells.reduce((a, n) => a + Number(n.getAttribute('data-count')), 0), heat.plotted);
}
function cycleEverything() {
  checkInvariants('initial');
  for (const hitter of ['all', 'near', 'far']) {
    for (const type of get('fType').children.map(b => b.getAttribute('data-v')).filter(Boolean)) {
      for (const est of [true, false]) {
        chip('fHitter', hitter).fire('click'); chip('fType', type).fire('click'); setEst(est);
        checkInvariants(`${hitter}/${type}/${est}`);
      }
    }
  }
  chip('fHitter', 'all').fire('click'); chip('fType', 'all').fire('click'); setEst(true);
  for (const tr of [...pointRowsDom()]) {
    tr.fire('click'); checkInvariants('selected ' + tr.getAttribute('data-point'));
    assert.equal(get('seq').hidden, false);
    pointRowsDom().find(r => r.getAttribute('data-point') === tr.getAttribute('data-point')).fire('click');
    assert.equal(evaluate('state.sel'), null); assert.equal(get('seq').hidden, true);
  }
}

if (!pagePath) {
  // ---------------------------------------------------------------- fixture specifics
  assert.equal(evaluate('MD.shotsInView(DATA, state).length'), 6, 'six shots; the uncertain contact is not a shot');
  assert.match(get('tiles').textContent, /6shots in view · 1 contacts in the table are not shots/);
  assert.match(get('tiles').textContent, /3 \/ 6landings from confirmed bounces \(coverage, not accuracy\)/);
  assert.deepEqual(marks().map(n => n.getAttribute('data-source')).sort(), ['estimated', 'human_confirmed', 'observed', 'observed']);
  assert.equal(marks().find(n => n.getAttribute('data-source') === 'estimated').getAttribute('fill'), 'var(--panel)', 'estimated = hollow');
  assert.equal(marks().find(n => n.getAttribute('data-source') === 'human_confirmed').getAttribute('fill'), 'var(--far)', 'confirmed = filled in the hitter colour');
  assert.match(get('courtCount').textContent, /^4 plotted · 2 unknown \(not plotted\)$/);
  // Untrusted basis text stays text (in an SVG <title>).
  assert.ok(all(get('court'), n => n.tagName === 'title' && n.textContent.includes('<img src=x onerror=alert(1)>')).length === 1);

  // Hide estimated: candidate shots and estimated landings drop out, and every chart follows.
  setEst(false);
  assert.deepEqual(evaluate('MD.shotsInView(DATA, state).map(r => r.shot_id)'), ['a', 'b', 'c', 'd']);
  assert.match(get('courtCount').textContent, /^2 plotted · 1 unknown \(not plotted\) · 1 estimated hidden$/);
  assert.equal(evaluate('MD.heatGrid(DATA, state).plotted'), 4);
  assert.equal(pointRowsDom().length, 1, 'only p1 has a non-estimated shot');
  assert.ok(!get('shotsBody').textContent.includes('km/h'), 'estimated speed hidden with estimated values');
  assert.equal(JSON.parse(store.get('matchdesk.filters')).showEst, false, 'filters persist');
  setEst(true);

  // Hitter filter.
  chip('fHitter', 'far').fire('click');
  assert.deepEqual(evaluate('MD.shotsInView(DATA, state).map(r => r.shot_id)'), ['b', 'd', 'e']);
  assert.equal(chip('fHitter', 'far').getAttribute('aria-pressed'), 'true');
  assert.equal(chip('fHitter', 'all').getAttribute('aria-pressed'), 'false');
  assert.equal(marks().length, 2);
  chip('fHitter', 'near').fire('click');
  assert.deepEqual(pointRowsDom().map(r => r.getAttribute('data-point')), ['p1'], 'p2 has only a far shot and a near contact');
  chip('fHitter', 'all').fire('click');

  // Shot-type filter; the shot mix dims other types instead of dropping them.
  chip('fType', 'forehand').fire('click');
  assert.deepEqual(evaluate('MD.shotsInView(DATA, state).map(r => r.shot_id)'), ['b', 'd', 'g']);
  const mixRows = get('mix').children;
  assert.deepEqual(mixRows.map(r => r.getAttribute('data-type')), ['serve', 'forehand', 'backhand']);
  assert.ok(all(mixRows[0], n => (n.getAttribute('class') || '').includes('dim')).length > 0);
  assert.equal(all(mixRows[1], n => (n.getAttribute('class') || '').includes('dim')).length, 0);
  assert.deepEqual(evaluate('MD.shotMix(DATA, state).rows.map(r => [r.type, r.counts.near, r.counts.far])'),
    [['serve', 1, 1], ['forehand', 1, 2], ['backhand', 1, 0]]);
  chip('fType', 'unknown').fire('click');
  assert.equal(evaluate('MD.shotsInView(DATA, state).length'), 0);
  assert.match(get('courtNote').textContent, /No shots match/);
  chip('fType', 'all').fire('click');

  // Points list: score, server, winner only from labels; confirmed landings x/n.
  const rows = pointRowsDom();
  assert.equal(rows.length, 2);
  assert.equal(rows[0].children[2].textContent, '15-0');
  assert.equal(rows[0].children[5].textContent, 'Near');
  assert.equal(rows[0].children[6].textContent, '1/4 (+1 observed)');
  assert.equal(rows[1].children[2].textContent, '—', 'score not labelled');
  assert.equal(rows[1].children[5].textContent, 'not labelled', 'winner never inferred');
  assert.match(rows[1].children[4].textContent, /^1estimated$/, 'candidate shot count is tagged estimated');
  assert.match(get('strip').textContent, /Near won 1.*Far won 0.*1 of 2 labelled points · 1 with winner not labelled/);

  // Selecting a point draws its shots in order, contact -> landing, and links each shot to the replay.
  rows[0].fire('click');
  assert.equal(evaluate('state.sel'), 'p1');
  const lines = all(get('court'), n => n.tagName === 'line' && (n.getAttribute('marker-end') || '').startsWith('url(#arr-'));
  assert.equal(lines.length, 3, 'a, b, c have contact and landing; d has no landing');
  assert.equal(lines[2].getAttribute('stroke-dasharray'), '5 3', 'estimated landing drawn dashed');
  const seqItems = get('seq').children[1].children;
  assert.deepEqual(seqItems.map(li => li.getAttribute('data-shot')), ['a', 'b', 'c', 'd']);
  assert.match(seqItems[0].textContent, /Near serve.*cross court, mid.*~72\.5 km\/h \(estimated\)/);
  assert.match(seqItems[2].textContent, /called out/);
  assert.match(seqItems[3].textContent, /landing unknown/);
  assert.deepEqual(links(get('seq')).map(a => a.getAttribute('href')), ['replay.html#frame=10', 'replay.html#frame=40', 'replay.html#frame=70', 'replay.html#frame=100']);
  assert.match(get('seq').textContent, /Won by Near \(labelled\)/);
  assert.match(get('seq').textContent, /does not seek/);
  assert.equal(get('clearSel').hidden, false);
  get('clearSel').fire('click');
  assert.equal(evaluate('state.sel'), null);
  rows[1].fire('keydown', {key: 'Enter'});
  assert.equal(evaluate('state.sel'), 'p2');
  assert.match(get('seq').textContent, /Winner not labelled/);
  chip('fHitter', 'all').fire('click');
  assert.equal(evaluate('state.sel'), null, 'changing a filter clears the selection');

  // Rally length and patterns: sample sizes everywhere; won only from labelled winner + server.
  const hist = evaluate('MD.rallyHistogram(DATA, state)');
  assert.deepEqual([hist.buckets[0].n, hist.buckets[3].n, hist.estimated], [1, 1, 1]);
  assert.match(get('rallyNote').textContent, /n = 2 of 2 labelled points; 1 counts are unreviewed shot candidates \(estimated\)/);
  const pat = evaluate('MD.patterns(DATA, state)');
  assert.deepEqual(pat.serve.map(r => [r.direction, r.points, r.server_won, r.labelled]), [['cross_court', 2, 1, 1]]);
  assert.deepEqual(pat.lengths.map(r => [r.points, r.labelled, r.server_won]), [[1, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]]);
  assert.match(get('serveBody').textContent, /cross court21 of 1 \(100%\)2\.5 \(n = 2\)/);
  assert.match(get('lenBody').textContent, /1–3 shots1n = 0n = 0/);
  assert.deepEqual(pat.depth.map(r => [r.depth, r.near, r.far]), [['short', 0, 1], ['mid', 1, 0], ['deep', 1, 1], [null, 1, 1]]);
  setEst(false);
  assert.equal(evaluate('MD.patterns(DATA, state).serve[0].direction'), null, 'estimated zones hidden');
  setEst(true);

  // Heatmap: positions outside the drawn area are counted, not clamped.
  const heat = evaluate('MD.heatGrid(DATA, state)');
  assert.deepEqual([heat.plotted, heat.outside, heat.unknown], [5, 1, 0]);
  assert.match(get('heatNote').textContent, /n = 5 of 6 shots in view; 1 outside the drawn area/);

  // Provenance and limits are on the page.
  assert.match(get('prov').textContent, /Replay: fixture-replay.*Calibration: reviewed.*Labels: labels.json by test/);
  assert.match(get('limits').textContent, /Spin and RPM are never shown/);

  // Theme toggle: auto -> light -> dark -> auto, remembered.
  assert.equal(root.getAttribute('data-theme'), null);
  get('theme').fire('click'); assert.equal(root.getAttribute('data-theme'), 'light');
  get('theme').fire('click'); assert.equal(root.getAttribute('data-theme'), 'dark');
  assert.equal(store.get('matchdesk.theme'), 'dark');
  get('theme').fire('click'); assert.equal(root.getAttribute('data-theme'), null);

  // A blocked localStorage must not break filtering.
  context.localStorage.setItem = () => { throw Error('disabled'); };
  chip('fHitter', 'near').fire('click');
  assert.equal(evaluate('state.hitter'), 'near');
  chip('fHitter', 'all').fire('click');

  // No replay link -> no anchors.
  run('DATA.replay_link = {href: null, status: "replay folder not found", seek: null}; renderAll();');
  assert.equal(links(get('shotsBody')).length, 0);
  assert.match(get('shotsSummary').textContent, /no replay links: replay folder not found/);
}

cycleEverything();

if (synthetic) {
  // tests/test_match_dashboard.py's synthetic table (built through shot_table).
  assert.deepEqual(marks().map(n => n.getAttribute('data-source')).sort(), ['estimated', 'human_confirmed', 'observed', 'observed']);
  assert.equal(pointRowsDom().length, 2);
  assert.ok(links(get('shotsBody')).every(a => /^\.\.\/replay\/replay\.html#frame=\d+$/.test(a.getAttribute('href'))));
  assert.equal(links(get('shotsBody')).length, 6);
}
if (pagePath && !data.points.length) {
  assert.match(get('ptsNote').textContent, /No labelled points/);
  assert.match(get('strip').textContent, /No labelled points/);
  assert.match(get('rally').textContent, /No labelled points/);
}
console.log(`Match dashboard UI logic checks passed (${pagePath ? path.basename(pagePath) : 'fixture'}; fake DOM, not a browser render).`);
