// Pure logic checks with a fake DOM; not a browser visual/playback test.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2]||path.join(__dirname,'../runs/court-bounce-calibrated/calibration.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const data=JSON.parse(scripts.find(s=>s[1].includes('application/json'))[2]);
class Element{
 constructor(id){this.id=id;this.children=[];this.handlers={};this.width=1280;this.height=720;this.checked=false;this.value='';this.textContent='';this.draws=[];}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}click(){return this.onclick?.();}
 getBoundingClientRect(){return {left:0,top:0,width:640,height:360};}
 getContext(){const out={};for(const fn of ['drawImage','clearRect','beginPath','moveTo','lineTo','stroke','arc','fillText'])out[fn]=(...args)=>this.draws.push({fn,args,stroke:out.strokeStyle});return out;}
}
class ImageStub{
 constructor(){this.complete=false;}
 set src(v){this._src=v;this.currentSrc=v;this.complete=true;this.onload?.();}get src(){return this._src;}
}
const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);};
get('calibration-data').textContent=JSON.stringify(data);get('showNew').checked=true;get('showBall').checked=true;
const saved=new Map();let download;
const ctx=vm.createContext({document:{getElementById:get,createElement:id=>new Element(id)},Image:ImageStub,
 localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},Blob,
 URL:{createObjectURL:b=>{download=b;return 'fake';},revokeObjectURL(){}},setTimeout:f=>f(),console});
vm.runInContext(scripts.find(s=>!s[1].includes('application/json'))[2],ctx);
const run=s=>vm.runInContext(s,ctx);
assert.equal(get('landmark').children.length,15);assert.equal(get('event').children.length,9);
assert.equal(run('loadedFrame'),122);assert.equal(run('draft.calibration_status'),'draft');
assert.ok(get('calibrationCanvas').draws.some(d=>d.fn==='stroke'&&d.stroke==='#7aff91'));
get('event').value='bounce_candidate-000519';get('event').onchange();
assert.equal(run('frame'),518);assert.equal(run('loadedFrame'),518);assert.match(get('frameInfo').textContent,/source frame 5258/);
get('previous').click();assert.equal(run('frame'),517);get('next').click();assert.equal(run('frame'),518);
get('confirm').click();assert.equal(run('edit().status'),'confirmed');
assert.equal(data.draft.bounce_edits[1].status,'unreviewed','source payload must not be mutated');
get('pickLanding').checked=true;get('bounceCanvas').onclick({clientX:300,clientY:80});
assert.equal(run('JSON.stringify(edit().landing_pixel)'),'[600,160]');assert.equal(run('edit().status'),'unreviewed');
get('confirm').click();get('next').click();get('useFrame').click();
assert.equal(run('edit().frame'),519);assert.equal(run('edit().landing_pixel'),null);assert.equal(run('edit().status'),'unreviewed');
get('undo').click();assert.equal(run('edit().frame'),518);assert.equal(run('edit().status'),'confirmed');
run('loadedFrame=null');get('confirm').click();assert.match(get('notice').textContent,/Wait/);run('loadFrame(518)');
get('unsure').click();assert.equal(run('edit().status'),'uncertain');
get('reject').click();assert.equal(run('edit().status'),'rejected');
get('checkedCourt').checked=true;get('checkedCourt').onchange();assert.equal(run('draft.calibration_status'),'reviewed');
get('calibrationCanvas').draws=[];get('calibrationCanvas').onclick({clientX:5,clientY:260});
assert.equal(run('dirty'),true);assert.equal(run('draft.calibration_status'),'draft');
assert.equal(get('checkedCourt').checked,false);
assert.ok(!get('calibrationCanvas').draws.some(d=>d.fn==='stroke'&&d.stroke==='#7aff91'),'stale fitted curve must be hidden');
get('undo').click();assert.equal(run('dirty'),false);
assert.equal(run('validSaved({...draft,run_id:"wrong"})'),false);
get('export').click();
(async()=>{const out=JSON.parse(await download.text());assert.equal(out.run_id,data.draft.run_id);
 assert.equal(out.bounce_edits.find(e=>e.id==='bounce_candidate-000519').status,'rejected');
 assert.equal(out.calibration_status,'reviewed');
 console.log('Calibration UI logic checks passed (fake DOM; no actual browser verification).');
})().catch(e=>{console.error(e);process.exitCode=1;});
