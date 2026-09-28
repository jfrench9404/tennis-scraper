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
 addEventListener(type,handler){this.handlers[type]=handler;}setAttribute(name,value){this[name]=String(value);}click(){return this.onclick?.();}
 getContext(){const out={};for(const name of ['beginPath','moveTo','lineTo','stroke','fill','arc','fillText','setLineDash','closePath','clearRect'])out[name]=(...args)=>this.draws.push({fn:name,args,stroke:out.strokeStyle,fill:out.fillStyle,width:out.lineWidth,font:out.font});return out;}
 pause(){this.paused=true;this.handlers.pause?.();}play(){this.paused=false;this.handlers.play?.();return Promise.resolve();}setPointerCapture(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
get('replay-data').textContent=JSON.stringify(payload);get('speed').value='.5';const saved=new Map();let download;
const docHandlers={};
const context=vm.createContext({document:{getElementById:get,createElement:name=>new Element(name),addEventListener:(type,handler)=>{docHandlers[type]=handler;}},
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
// Keyboard shortcuts (issue #5): navigation/display only, ignored while typing.
{
 const press=(key,extra={})=>{const e={key,code:key===' '?'Space':'',shiftKey:false,ctrlKey:false,metaKey:false,altKey:false,repeat:false,defaultPrevented:false,target:{tagName:'BODY'},prevented:false,preventDefault(){this.prevented=true;},...extra};docHandlers.keydown(e);return e;};
 assert.equal(typeof docHandlers.keydown,'function','page registers one document keydown handler');
 const reviewState=()=>run('JSON.stringify([D.events.map(e=>e.status),corrections,contactLabels,handCorrections])'),reviewBefore=reviewState();
 // Space: play/pause, and the page does not also scroll.
 assert.equal(get('video').paused,true);let e=press(' ');assert.equal(get('video').paused,false);assert.equal(e.prevented,true);
 press(' ');assert.equal(get('video').paused,true);
 assert.equal(press(' ',{repeat:true}).prevented,false,'held Space does not re-toggle playback');assert.equal(get('video').paused,true);
 // Arrows: one frame; Shift+arrow one second (round(fps) frames); clamped to the clip.
 const second=Math.round(payload.report.fps);run('seek(100)');
 press('ArrowRight');assert.equal(run('frame'),101);press('ArrowLeft');assert.equal(run('frame'),100);
 press('ArrowRight',{shiftKey:true});assert.equal(run('frame'),100+second);press('ArrowLeft',{shiftKey:true});assert.equal(run('frame'),100);
 press('ArrowRight',{repeat:true});assert.equal(run('frame'),101,'held arrows keep stepping');
 get('video').play();press('ArrowRight');assert.equal(get('video').paused,true,'frame stepping pauses playback');
 run('seek(0)');press('ArrowLeft');assert.equal(run('frame'),0);press('ArrowLeft',{shiftKey:true});assert.equal(run('frame'),0);
 run(`seek(${payload.report.frames-1})`);press('ArrowRight',{shiftKey:true});assert.equal(run('frame'),payload.report.frames-1);
 // Focus in form controls (and browser/OS modifier chords) leaves every key alone.
 run('seek(100)');
 for(const tagName of ['INPUT','input','SELECT','TEXTAREA'])for(const key of [' ','ArrowRight','n','p','b','m','w','?']){
  const helpHidden=get('shortcutHelp').hidden,renderer=get('bodyRenderer').value,wire=get('meshWireframe').checked,picked=run('selected?.event_id??selectedBounce?.id??null');
  e=press(key,{target:{tagName}});assert.equal(e.prevented,false,`${key} in ${tagName}`);
  assert.equal(get('video').paused,true);assert.equal(run('frame'),100);assert.equal(get('shortcutHelp').hidden,helpHidden);
  assert.equal(get('bodyRenderer').value,renderer);assert.equal(get('meshWireframe').checked,wire);assert.equal(run('selected?.event_id??selectedBounce?.id??null'),picked);
 }
 assert.equal(press('ArrowRight',{target:{tagName:'DIV',isContentEditable:true}}).prevented,false);
 for(const mod of ['ctrlKey','metaKey','altKey'])assert.equal(press('ArrowRight',{[mod]:true}).prevented,false);
 assert.equal(run('frame'),100);
 // N/P walk the Activity list in time order and respect its filter.
 const states=[...run('D.shots.shots.map(s=>s.action_state??null)')];
 run("D.shots.shots.forEach((s,i)=>{s.action_state=['shot_candidate','uncertain_contact','no_shot_candidate','confirmed_shot'][i%4];})");
 const ids=filter=>{get('activityFilter').value=filter;get('activityFilter').onchange();return [...run('D.shots.shots.filter(activityVisible).sort((a,b)=>shotFrame(a)-shotFrame(b)).map(s=>s.event_id)')];};
 for(const filter of ['shots','uncertain','all']){
  const expected=ids(filter);assert.ok(expected.length,`fixture has ${filter} events`);
  run('selected=null;selectedBounce=null;seek(0)');const walked=[];
  for(let i=0;i<expected.length;i++){assert.equal(press('n').prevented,true);walked.push(run('selected.event_id'));assert.equal(run('frame'),run('shotFrame(selected)'));}
  assert.deepEqual(walked,expected,`N follows the ${filter} list`);
  press('n');assert.equal(run('selected.event_id'),expected.at(-1),'N stops at the last event (no wrap)');assert.match(get('notice').textContent,/No later event/);
  const back=[expected.at(-1)];for(let i=1;i<expected.length;i++){press('p');back.push(run('selected.event_id'));}
  assert.deepEqual(back,[...expected].reverse(),`P walks back through the ${filter} list`);
  press('p');assert.equal(run('selected.event_id'),expected[0]);assert.match(get('notice').textContent,/No earlier event/);
 }
 const uncertainOnly=ids('uncertain'),shotsOnly=ids('shots');assert.ok(!shotsOnly.some(id=>uncertainOnly.includes(id)));
 // A selected event outside the current filter: N/P fall back to the current frame.
 run(`selectShot(D.shots.shots.find(s=>s.event_id===${JSON.stringify(uncertainOnly[0])}))`);const from=run('frame');
 press('n');assert.ok(shotsOnly.includes(run('selected.event_id')));assert.ok(run('shotFrame(selected)')>from);
 // With nothing selected, N/P start from the current frame.
 const frames=shotsOnly.map(id=>run(`shotFrame(D.shots.shots.find(s=>s.event_id===${JSON.stringify(id)}))`));
 run(`selected=null;seek(${frames[0]})`);press('n');assert.equal(run('selected.event_id'),shotsOnly.find((id,i)=>frames[i]>frames[0]));
 run(`selected=null;seek(${frames[0]+1})`);press('p');assert.equal(run('selected.event_id'),shotsOnly[0]);
 run(`D.shots.shots.forEach((s,i)=>{const v=${JSON.stringify(states)}[i];if(v===null)delete s.action_state;else s.action_state=v;})`);ids('shots');
 // B walks bounces/candidates forward in time, never wraps, and never confirms them.
 const bounceIds=[...run('[...bounces].sort((a,b)=>a.frame-b.frame).map(e=>e.id)')];assert.ok(bounceIds.length);
 run('selected=null;selectedBounce=null;seek(0)');const seen=[];
 for(const id of bounceIds){press('b');seen.push(run('selectedBounce.id'));assert.equal(run('frame'),run('selectedBounce.frame'));assert.equal(run('selected'),null);}
 assert.deepEqual(seen,bounceIds);press('b');assert.equal(run('selectedBounce.id'),bounceIds.at(-1));assert.match(get('notice').textContent,/No later bounce/);
 // M/W: meshes vs lines, wireframe. Without a WebGL mesh renderer they only explain why.
 get('bodyRenderer').value='mesh';get('meshWireframe').checked=false;
 press('m');assert.equal(get('bodyRenderer').value,'mesh');assert.match(get('notice').textContent,/unavailable/);
 press('w');assert.equal(get('meshWireframe').checked,false);assert.match(get('notice').textContent,/unavailable/);
 let changes=0;get('bodyRenderer').onchange=()=>changes++;get('meshWireframe').onchange=()=>changes++;
 run("meshRenderer={enabled:()=>$('bodyRenderer').value==='mesh',render(){}}");
 press('m');assert.equal(get('bodyRenderer').value,'lines');assert.equal(changes,1);assert.match(get('notice').textContent,/Bodies: pose lines/);
 press('w');assert.equal(get('meshWireframe').checked,true);assert.equal(changes,2);assert.match(get('notice').textContent,/wireframe on.*press M/);
 press('m');assert.equal(get('bodyRenderer').value,'mesh');assert.match(get('notice').textContent,/template shape over observed joints/);
 press('w');assert.equal(get('meshWireframe').checked,false);assert.equal(changes,4);
 assert.equal(press('m',{repeat:true}).prevented,false);assert.equal(get('bodyRenderer').value,'mesh');
 get('bodyRenderer').disabled=true;press('m');assert.equal(get('bodyRenderer').value,'mesh');get('bodyRenderer').disabled=false;
 run('meshRenderer=null');get('bodyRenderer').onchange=null;get('meshWireframe').onchange=null;
 // ? toggles the help panel (as does its button); Esc closes it.
 assert.match(template,/<section id="shortcutHelp"[^>]*\bhidden\b/,'help panel starts closed');get('shortcutHelp').hidden=true;// fake DOM does not parse markup attributes
 assert.equal(press('?',{shiftKey:true}).prevented,true);
 assert.equal(get('shortcutHelp').hidden,false);assert.equal(get('shortcutsButton')['aria-expanded'],'true');
 press('?',{shiftKey:true});assert.equal(get('shortcutHelp').hidden,true);assert.equal(get('shortcutsButton')['aria-expanded'],'false');
 get('shortcutsButton').click();assert.equal(get('shortcutHelp').hidden,false);press('Escape');assert.equal(get('shortcutHelp').hidden,true);
 press('?');assert.equal(get('shortcutHelp').hidden,false);get('closeShortcuts').click();assert.equal(get('shortcutHelp').hidden,true);assert.equal(get('shortcutsButton')['aria-expanded'],'false');
 assert.equal(press('Escape').prevented,false,'Esc is left alone when the panel is closed');
 assert.equal(press('x').prevented,false,'unbound keys are left alone');
 assert.equal(reviewState(),reviewBefore,'shortcuts never change review data');
 get('shotList').children[0].click();
}
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
