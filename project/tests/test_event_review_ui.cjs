// Logic-only unit test with a fake DOM. Does NOT launch/control a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(process.argv[2] || path.join(__dirname, '../runs/rally-event-review-verified/review.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const data = JSON.parse(scripts.find(s => s[1].includes('application/json'))[2]);
const code = scripts.find(s => !s[1].includes('application/json'))[2];
class Element {
  constructor(id) { this.id = id; this.children = []; this.handlers = {}; this.textContent = ''; this.value = ''; this.currentTime = 0; this.readyState = 1; this.width = 1000; this.hidden = false; }
  get value() { return this._value; }
  set value(value) { this._value = String(value); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, handler) { this.handlers[type] = handler; }
  getContext() { return {fillRect(){}, clearRect(){}}; }
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
  load() { this.handlers.loadedmetadata?.(); }
  click() { this.onclick?.(); }
}
const elements = new Map();
const get = id => { if(!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); };
get('review-data').textContent = JSON.stringify(data);
get('speed').value = '.5';
const saved = new Map();
let exported;
const context = vm.createContext({document:{getElementById:get,createElement:tag=>new Element(tag)},
  window:{addEventListener(){}},localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},
  Blob,URL:{createObjectURL:blob=>{exported=blob;return 'fake-download';},revokeObjectURL(){}},setTimeout:f=>f(),console});
vm.runInContext(code, context); // Includes syntax checking and page initialization.
const evaluate = expr => vm.runInContext(expr, context);
const first = data.events[0];
assert.ok(first, 'Need candidate fixtures');
assert.equal(get('events').children.length, data.events.length);
assert.equal(evaluate('selected.id'), first.id);
assert.equal(evaluate('frameNow()'), first.frame);
get('nextFrame').click(); assert.equal(evaluate('frameNow()'), first.frame+1);
get('previousFrame').click(); assert.equal(evaluate('frameNow()'), first.frame);
get('kind').value='hit';get('player').value='near';get('confirm').click();
assert.equal(evaluate('labels[selected.id].status'), 'confirmed');
assert.equal(JSON.parse([...saved.values()][0]).labels[0].source_frame, data.report.source_start_frame+first.frame);
get('eventFrame').value='';get('confirm').click();assert.match(get('notice').textContent,/whole frame/);
get('eventFrame').value='-1';get('confirm').click();assert.match(get('notice').textContent,/whole frame/);
get('eventFrame').value=String(first.frame+1);get('eventFrame').handlers.change();
assert.equal(evaluate('labels[selected.id].status'), 'unreviewed');
get('player').value='';get('player').handlers.change();
assert.equal(evaluate('labels[selected.id].player_id'), null);
assert.match(get('events').children[0].children[0].textContent, /—$/);
get('notes').value='<script>untrusted note</script>';get('notes').handlers.input();get('reject').click();
assert.equal(evaluate('labels[selected.id].status'), 'rejected');
get('kind').value='uncertain';get('confirm').click();assert.match(get('notice').textContent,/Choose Hit or Bounce/);
get('add').click(); assert.equal(evaluate('manual.length'),1);
get('uncertain').click();assert.equal(evaluate('labels[selected.id].status'),'uncertain');
assert.equal(get('clip').hidden,true);
get('raw').checked=true;const oldTime=get('video').currentTime;get('raw').onchange();
assert.equal(get('video').src,'source.mp4');assert.equal(get('video').currentTime,oldTime);
get('replay').click();assert.equal(get('video').paused,false);
evaluate('video.currentTime=stopAt+.01');get('video').handlers.timeupdate();assert.equal(get('video').paused,true);
get('export').click();
(async()=>{
  const payload=JSON.parse(await exported.text());
  assert.equal(payload.run_id,data.report.run_id);assert.equal(payload.manual_events.length,1);
  context.fixture=payload;
  assert.equal(evaluate('Object.keys(validateFile(fixture).labels).length'),2);
  payload.run_id='wrong-run';assert.throws(()=>evaluate('validateFile(fixture)'),/exact review run/);payload.run_id=data.report.run_id;
  payload.labels[0].frame=data.report.frames;assert.throws(()=>evaluate('validateFile(fixture)'),/Invalid review label/);
  payload.labels[0].frame=first.frame;payload.labels[0].status='made-up';assert.throws(()=>evaluate('validateFile(fixture)'),/Invalid review label/);
  payload.labels[0].status='rejected';payload.labels.push(payload.labels[0]);assert.throws(()=>evaluate('validateFile(fixture)'),/Invalid review label/);payload.labels.pop();
  payload.labels[0].notes='Restored test label';
  await get('importFile').onchange({target:{files:[{size:100,text:async()=>JSON.stringify(payload)}],value:'fake'}});
  assert.equal(evaluate(`labels[${JSON.stringify(first.id)}].notes`),'Restored test label');
  // A blocked localStorage must not prevent export or label changes.
  context.localStorage.setItem=()=>{throw Error('disabled');};get('uncertain').click();
  assert.match(get('notice').textContent,/autosave unavailable/);get('export').click();assert.ok(exported);
  console.log('Review UI logic checks passed (fake DOM; not a visual/browser playback test).');
})().catch(error=>{console.error(error);process.exitCode=1;});
