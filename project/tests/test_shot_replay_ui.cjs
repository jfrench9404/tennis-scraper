// Logic/geometry checks only. Fake DOM; never opens a browser or changes labels.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2]||path.join(__dirname,'../runs/shot-replay-trajectories/replay.html'),'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)],payload=JSON.parse(scripts.find(s=>s[1].includes('application/json'))[2]);
const template=fs.readFileSync(path.join(__dirname,'../tennis_vision/shot_replay.html'),'utf8');
const code=[...template.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)].find(s=>!s[1].includes('application/json'))[2];
class Element{
 constructor(id){this.id=id;this.children=[];this.handlers={};this.value='';this.textContent='';this.width=1000;this.height=562;this.currentTime=0;this.readyState=1;this.paused=true;this.checked=true;this.hidden=false;this.draws=[];}
 get value(){return this._value;}set value(v){this._value=String(v);}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}
 addEventListener(type,handler){this.handlers[type]=handler;}click(){return this.onclick?.();}
 getContext(){const out={};for(const name of ['beginPath','moveTo','lineTo','stroke','fill','arc','fillText','setLineDash','closePath','clearRect'])out[name]=(...args)=>this.draws.push({fn:name,args,stroke:out.strokeStyle,fill:out.fillStyle,width:out.lineWidth,font:out.font});return out;}
 pause(){this.paused=true;this.handlers.pause?.();}play(){this.paused=false;this.handlers.play?.();return Promise.resolve();}setPointerCapture(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
get('replay-data').textContent=JSON.stringify(payload);get('speed').value='.5';const saved=new Map();let download;
const context=vm.createContext({document:{getElementById:get,createElement:name=>new Element(name)},
 localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},Blob,URL:{createObjectURL:b=>{download=b;return 'fake';},revokeObjectURL(){}},setTimeout:f=>f(),requestAnimationFrame(){},console});
vm.runInContext(code,context);const run=s=>vm.runInContext(s,context);
// Optional generated draw-command artifact for offline geometry inspection.
// This is not a browser screenshot or a browser-rendering verification.
if(process.argv[3]){get('world').draws=[];run('frame=Math.floor((previews.fits[0].start_frame+previews.fits[0].end_frame)/2);drawWorld()');fs.writeFileSync(process.argv[3],JSON.stringify({frame:run('frame'),commands:get('world').draws}));run('refresh()');}
assert.equal(get('shotList').children.length,payload.shots.shots.length);
assert.equal(run('frame'),payload.preview_flights.fits[0].start_frame);
assert.equal(get('pending').hidden,false);assert.match(get('ballState').textContent,/UNVALIDATED PREVIEW/);
assert.equal(get('trajectoryList').children.length,payload.preview_flights.fits.length+payload.flights.fits.length);
const modes=JSON.stringify(payload.events.map(e=>e.status));
get('world').draws=[];run('drawWorld()');
assert.ok(get('world').draws.some(d=>d.fn==='stroke'&&d.stroke==='#477b89'),'full estimated segment is drawn immediately');
get('fullTrajectory').checked=false;get('world').draws=[];get('fullTrajectory').onchange();
assert.ok(!get('world').draws.some(d=>d.fn==='stroke'&&d.stroke==='#477b89'));
get('fullTrajectory').checked=true;
get('ballMode').value='reviewed';get('ballMode').onchange();assert.equal(run('trajectoryAt(frame)'),null);
assert.match(get('ballState').textContent,/no confirmed bounce/);
get('ballMode').value='off';get('ballMode').onchange();assert.match(get('ballState').textContent,/hidden by display/);
get('ballMode').value='preview';get('ballMode').onchange();assert.equal(run('trajectoryAt(frame).kind'),'preview');
// Reviewed estimates take priority without replacing the preview data channel.
run('D.frames[frame].ball_3d={point_m:[0,0,1],fit_index:0};drawWorld()');
assert.equal(run('trajectoryAt(frame).kind'),'reviewed');run('D.frames[frame].ball_3d=null;drawWorld()');
get('trajectoryList').children.at(-1).click();assert.equal(run('frame'),payload.preview_flights.fits.at(-1).start_frame);
get('playTrajectory').click();assert.equal(get('video').paused,false);
run('video.currentTime=stopAt+.5');get('video').handlers.timeupdate();
assert.equal(get('video').paused,true);assert.equal(run('frame'),payload.preview_flights.fits.at(-1).end_frame);
assert.match(get('ballState').textContent,/UNVALIDATED PREVIEW/);
get('next').click();assert.equal(run('trajectoryAt(frame)'),null);assert.match(get('ballState').textContent,/No supported/);
assert.equal(run('JSON.stringify(D.events.map(e=>e.status))'),modes);
get('trajectoryList').children[0].click();
// Loading the local video later must honor the requested trajectory start.
get('video').readyState=0;get('video').currentTime=0;get('trajectoryList').children[0].click();
assert.notEqual(run('pendingSeek'),null);get('video').readyState=1;get('video').handlers.loadedmetadata();
assert.equal(run('pendingSeek'),null);assert.equal(run('frame'),payload.preview_flights.fits[0].start_frame);
const projected=run('project([0,0,0])');assert.ok(projected.every(Number.isFinite));assert.ok(projected[0]>=0&&projected[0]<=1000);
const before=run('frame');get('next').click();assert.equal(run('frame'),before+1);get('prev').click();assert.equal(run('frame'),before);
// Candidate display must not require (or manufacture) a confirmed physics fit.
assert.equal(get('bounceList').children.length,payload.events.filter(e=>e.type==='bounce'&&e.status!=='rejected').length);
get('bounceList').children[0].click();assert.equal(run('frame'),run('bounces[0].frame'));
assert.equal(get('correction').hidden,true);assert.equal(run('selected'),null);
assert.match(get('shotSummary').textContent,/not a verified bounce/);
assert.equal(run('D.flights.fits.length'),0);assert.equal(run('bounces[0].court_m'),null);
assert.ok(run('bounces[0].candidate_court_m'));
get('world').draws=[];run('drawWorld()');
assert.ok(get('world').draws.some(d=>d.fn==='fillText'&&String(d.args[0]).startsWith('bounce?')));
get('showCandidates').checked=false;get('world').draws=[];get('showCandidates').onchange();
assert.ok(!get('world').draws.some(d=>d.fn==='fillText'&&String(d.args[0]).startsWith('bounce?')));
get('showCandidates').checked=true;get('showCandidates').onchange();
get('showPose').checked=false;get('imageOverlay').draws=[];run('drawOverlay()');
assert.ok(get('imageOverlay').draws.some(d=>d.fn==='arc'&&d.args[2]===7),'2D ball remains when pose overlay is off');
assert.ok(get('imageOverlay').draws.some(d=>d.fn==='fillText'&&d.args[0]==='bounce candidate?'));
get('showPose').checked=true;get('shotList').children[0].click();
assert.equal(run('selectedBounce'),null);assert.equal(get('correction').hidden,false);
run('seek(0)');assert.match(get('playerState').textContent,/far: observed pose.*back-associated identity/);
get('shotList').children[0].click();
get('followFar').click();assert.equal(run('focus'),'far');assert.equal(run('distance'),10);
assert.ok(run('project(D.frames[frame].players.find(p=>p.identity_id==="far").feet_xyz_m)').every(Number.isFinite));
get('reset').click();assert.equal(run('focus'),null);
get('top').click();assert.equal(run('elevation'),1.5);get('reset').click();assert.equal(run('elevation'),.62);
get('world').handlers.pointerdown({clientX:100,clientY:100,pointerId:1});get('world').handlers.pointermove({clientX:120,clientY:110});assert.ok(run('yaw')>0);get('world').handlers.pointerup();assert.equal(run('drag'),null);
get('world').handlers.wheel({deltaY:100000,preventDefault(){}});assert.equal(run('distance'),70);
get('world').handlers.wheel({deltaY:-100000,preventDefault(){}});assert.equal(run('distance'),18);
get('reset').click();get('play').click();assert.equal(get('video').paused,false);get('play').click();assert.equal(get('video').paused,true);
get('shotType').value='not_a_shot';get('saveType').click();assert.equal(run('corrections[selected.event_id]'),'not_a_shot');
get('replay').click();assert.equal(get('video').paused,false);run('video.currentTime=stopAt+.05');get('video').handlers.timeupdate();assert.equal(get('video').paused,true);
get('export').click();
(async()=>{const exported=JSON.parse(await download.text());assert.equal(exported.package_id,payload.report.package_id);assert.equal(exported.corrections[0].shot_type,'not_a_shot');
 // Missing frames must clear current balls/players rather than reuse last state.
 run('D.frames[1].ball_3d={point_m:[0,0,1],fit_index:0}; D.frames[2].ball_3d=null; frame=1;drawWorld()');
 assert.match(get('ballState').textContent,/Reviewed-constraint estimate/);const clears=get('world').draws.filter(d=>d.fn==='clearRect').length;
 run('frame=2;drawWorld()');assert.match(get('ballState').textContent,/No supported/);assert.equal(get('world').draws.filter(d=>d.fn==='clearRect').length,clears+1);
 run('D.frames[2].players=[];drawWorld()');assert.ok(get('world').draws.length);
 context.localStorage.setItem=()=>{throw Error('disabled');};get('saveType').click();assert.match(get('notice').textContent,/save unavailable/);
 console.log('Shot replay JS/geometry checks passed (fake DOM, not a browser visual test).');
})().catch(e=>{console.error(e);process.exitCode=1;});
