// Pure logic checks with a fake DOM; not a browser visual/playback test.
// 1. The page given as argv[2] (a historic built desk) keeps working.
// 2. The current template, filled with the same data, behaves the same.
// 3. The current template in carried-draft mode: provenance + camera check shown,
//    nothing marked reviewed unless John ticks it, export keeps the new run id and
//    carries no bounce edits.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const historic=fs.readFileSync(process.argv[2]||path.join(__dirname,'../runs/court-bounce-calibrated/calibration.html'),'utf8');
const template=fs.readFileSync(path.join(__dirname,'../tennis_vision/calibration_review.html'),'utf8');
const scriptsOf=html=>[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const dataOf=html=>JSON.parse(scriptsOf(html).find(s=>s[1].includes('application/json'))[2]);
const fill=packet=>template.replace('__CALIBRATION_DATA__',()=>JSON.stringify(packet).replace(/</g,'\\u003c'));
class Element{
 constructor(id){this.id=id;this.children=[];this.handlers={};this.width=1280;this.height=720;this.checked=false;this.disabled=false;this.hidden=false;this.value='';this.textContent='';this.draws=[];}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}click(){return this.onclick?.();}
 get selectedOptions(){return this.children.filter(c=>c.value===this.value);}
 getBoundingClientRect(){return {left:0,top:0,width:640,height:360};}
 getContext(){const out={};for(const fn of ['drawImage','clearRect','beginPath','moveTo','lineTo','stroke','arc','fillText'])out[fn]=(...args)=>this.draws.push({fn,args,stroke:out.strokeStyle});return out;}
}
function load(html,{saved=new Map(),missing=()=>false}={}){
 const scripts=scriptsOf(html),data=dataOf(html);
 class ImageStub{
  constructor(){this.complete=false;}
  set src(v){this._src=v;this.currentSrc=v;this.complete=true;if(missing(v))this.onerror?.();else this.onload?.();}get src(){return this._src;}
 }
 const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);},created=[];
 get('calibration-data').textContent=JSON.stringify(data);get('showNew').checked=true;get('showBall').checked=true;
 const state={download:null,anchor:null};
 const ctx=vm.createContext({document:{getElementById:get,createElement:tag=>{const e=new Element(tag);created.push(e);if(tag==='a')state.anchor=e;return e;}},Image:ImageStub,
  localStorage:{getItem:k=>saved.get(k),setItem:(k,v)=>saved.set(k,v)},Blob,
  URL:{createObjectURL:b=>{state.download=b;return 'fake';},revokeObjectURL(){}},setTimeout:f=>f(),console});
 vm.runInContext(scripts.find(s=>!s[1].includes('application/json'))[2],ctx);
 const exported=async()=>({name:state.anchor.download,out:JSON.parse(await state.download.text())});
 return {data,get,run:s=>vm.runInContext(s,ctx),saved,exported,created};
}
const text=el=>[el.textContent,...el.children.map(text)].join(' ');

async function standardDesk(html,label){
 const {data,get,run,exported}=load(html);
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
 const {name,out}=await exported();assert.equal(name,'tennis-calibration-review.json');assert.equal(out.run_id,data.draft.run_id);
 assert.equal(out.bounce_edits.find(e=>e.id==='bounce_candidate-000519').status,'rejected');
 assert.equal(out.calibration_status,'reviewed');
 console.log(`Calibration UI logic checks passed: ${label}.`);
}

function carriedPacket(base){
 const file=path.join(__dirname,'../runs/claude-pipeline-test-30/calibration-draft.json');
 const draft=fs.existsSync(file)?JSON.parse(fs.readFileSync(file,'utf8')):{...base.draft,run_id:'f'.repeat(64),calibration_frame:0,calibration_status:'draft',
  landmark_source:'carried_from_reviewed_run_x_after_camera_check',bounce_edits:[],carried_from:{run_id:base.draft.run_id,calibration_frame:300,status_there:'reviewed'},
  camera_check:{samples:[{frame:4740,time_s:158,score:1,best_shift_px:[0,0]},{frame:4769,time_s:158.97,score:1,best_shift_px:[0,0]}],summary:{verdict:'consistent',median_score:1,min_score:1,reference_score:1,threshold:.85,failing_samples:[]},meaning:'Consistency, not accuracy.'}};
 const start=4740,samples=draft.camera_check.samples.map(s=>({...s,run_frame:s.frame-start,image:`frames/${String(s.frame-start).padStart(6,'0')}.jpg`}));
 // Same landmarks as the historic desk, so its fitted curves are the right ones for this logic check.
 assert.deepEqual(draft.landmarks,base.draft.landmarks);
 return {mode:'carried_draft',draft,report:base.report,fps:30,source_start_frame:start,frames:30,landmark_definitions:base.landmark_definitions,
  curves:base.curves,old_curves:base.old_curves,events:[],ball_pixels:[],images_available:true,replay_href:'../claude-pipeline-test-30/replay/replay.html',
  commands:{apply:'python -m tennis_vision.calibration_review --corrections "$env:USERPROFILE\\Downloads\\x.json" --output ("runs\\a-" + $`)',rebuild:'rebuild <cmd>'},
  provenance:{draft_file:'runs/claude-pipeline-test-30/calibration-draft.json',replay:'runs/claude-pipeline-test-30/replay',run:'runs/claude-longrun-dml-30',review:'runs/claude-pipeline-test-30/review',
   carried_from:draft.carried_from,landmark_source:draft.landmark_source,notes:draft.notes},
  camera_check:{summary:draft.camera_check.summary,meaning:draft.camera_check.meaning,samples}};
}

async function draftDesk(base){
 const packet=carriedPacket(base),html=fill(packet),runId=packet.draft.run_id,sourceRun=packet.draft.carried_from.run_id;
 const page=load(html),{get,run,data,exported}=page;
 assert.equal(data.draft.calibration_status,'draft');
 // Opens with provenance and camera-check samples; bounce desk hidden.
 assert.equal(run('DRAFT'),true);assert.equal(get('provenance').hidden,false);assert.equal(get('bouncePanel').hidden,true);
 assert.equal(get('applyStandard').hidden,true);assert.equal(get('applyDraft').hidden,false);
 assert.match(get('title').textContent,/carried/i);
 const provenance=text(get('provenanceList'));
 assert.ok(provenance.includes(sourceRun),'source run id shown');assert.ok(provenance.includes(runId),'this run id shown');
 assert.equal(get('checkRows').children.length,packet.camera_check.samples.length);
 assert.ok(text(get('checkRows')).includes('4769'));assert.match(get('checkSummary').textContent,/consistent/);
 assert.match(get('checkMeaning').textContent,/not accuracy/);
 assert.equal(get('viewFrame').children.length,1+packet.camera_check.samples.length);
 assert.equal(get('applyCommand').textContent,packet.commands.apply,'commands shown verbatim');
 assert.equal(get('replayLink').href,packet.replay_href);
 // Nothing reviewed automatically.
 assert.equal(run('draft.calibration_status'),'draft');assert.equal(get('checkedCourt').checked,false);
 assert.equal(get('checkedCourt').disabled,false,'enabled once the frame image loaded');
 assert.match(get('calibrationState').textContent,/draft, not reviewed for this run/);
 assert.ok(get('calibrationCanvas').draws.some(d=>d.fn==='stroke'&&d.stroke==='#7aff91'),'fitted lines shown');
 assert.equal(run('selected'),null);
 get('export').click();
 let {name,out}=await exported();
 assert.equal(name,`tennis-calibration-draft-${runId.slice(0,16)}.json`);
 assert.equal(out.calibration_status,'draft');assert.equal(out.run_id,runId);assert.deepEqual(out.bounce_edits,[]);
 assert.deepEqual(out.landmarks,packet.draft.landmarks);assert.equal(out.landmark_source,packet.draft.landmark_source);
 // Camera-check frames can be shown behind the fitted lines.
 const row=get('checkRows').children[1],show=row.children.at(-1).children[0];
 get('calibrationCanvas').draws=[];show.click();
 assert.equal(run('baseImage.src'),packet.camera_check.samples[1].image);
 assert.match(get('calibrationState').textContent,/camera-check frame 29/);
 assert.ok(get('calibrationCanvas').draws.some(d=>d.fn==='stroke'&&d.stroke==='#7aff91'));
 // John moves a landmark, then explicitly marks it reviewed.
 get('landmark').value='far_service_centre';get('calibrationCanvas').onclick({clientX:302,clientY:79});
 assert.equal(run('dirty'),true);assert.equal(run('JSON.stringify(draft.landmarks.far_service_centre)'),'[604,158]');
 assert.equal(run('draft.calibration_status'),'draft');assert.match(get('calibrationState').textContent,/Landmarks moved/);
 get('checkedCourt').checked=true;get('checkedCourt').onchange();
 assert.equal(run('draft.calibration_status'),'reviewed');assert.match(get('calibrationState').textContent,/marked reviewed for this run by you/);
 get('export').click();({name,out}=await exported());
 assert.equal(name,`tennis-calibration-reviewed-${runId.slice(0,16)}.json`);
 assert.equal(out.calibration_status,'reviewed');assert.equal(out.run_id,runId,'keeps the new run id');
 assert.notEqual(out.run_id,sourceRun);assert.deepEqual(out.bounce_edits,[],'no bounce edits');
 assert.equal(out.landmark_source,'user_adjusted_landmarks');assert.deepEqual(out.carried_from,packet.draft.carried_from);
 assert.deepEqual(out.landmarks.far_service_centre,[604,158]);assert.match(get('notice').textContent,/reviewed for this run/);
 assert.equal(data.draft.calibration_status,'draft','source payload not mutated');
 // A later move clears the mark again; undo returns to the ticked state only through John's own action.
 get('calibrationCanvas').onclick({clientX:303,clientY:79});
 assert.equal(run('draft.calibration_status'),'draft');assert.equal(get('checkedCourt').checked,false);
 get('undo').click();assert.equal(run('draft.calibration_status'),'reviewed');
 // Bounce edits never pass validation or export in draft mode.
 assert.equal(run('validSaved({...draft,bounce_edits:[{id:"bounce_candidate-000519",frame:1,frame_range:[0,2],status:"confirmed",landing_pixel:null}]})'),false);
 assert.equal(run('validSaved({...draft,run_id:sourceRunId})'.replace('sourceRunId',JSON.stringify(sourceRun))),false);
 run('draft.bounce_edits=[{id:"x"}]');get('export').click();assert.match(get('notice').textContent,/Invalid review state/);run('draft.bounce_edits=[]');

 // Reload: landmark edits restore, the review mark never does.
 const again=load(html,{saved:page.saved});
 assert.equal(again.run('JSON.stringify(draft.landmarks.far_service_centre)'),'[604,158]');
 assert.equal(again.run('draft.calibration_status'),'draft');assert.equal(again.get('checkedCourt').checked,false);
 assert.match(again.get('notice').textContent,/never restored/);
 // A tampered saved state with bounce edits is ignored.
 const key=[...page.saved.keys()][0],tampered=new Map([[key,JSON.stringify({...JSON.parse(page.saved.get(key)),bounce_edits:[{id:'bounce_candidate-000519',frame:1,frame_range:[0,2],status:'confirmed',landing_pixel:null}]})]]);
 assert.deepEqual(JSON.parse(load(html,{saved:tampered}).run('JSON.stringify(draft.landmarks)')),packet.draft.landmarks);
 // Standard-desk autosave keys never collide with the draft desk.
 assert.ok(key.startsWith('tennis-court-review-v1-carried-draft-'+runId));

 // Without a frame image nothing can be marked reviewed or moved.
 const blind=load(html,{missing:src=>src.endsWith('.png')||src.endsWith('.jpg')});
 assert.equal(blind.get('checkedCourt').disabled,true);assert.match(blind.get('notice').textContent,/cannot be marked reviewed/);
 blind.get('checkedCourt').checked=true;blind.get('checkedCourt').onchange();
 assert.equal(blind.run('draft.calibration_status'),'draft');assert.equal(blind.get('checkedCourt').checked,false);
 blind.get('calibrationCanvas').onclick({clientX:5,clientY:260});assert.equal(blind.run('dirty'),false);
 blind.run('draft.calibration_status="reviewed"');blind.get('export').click();
 assert.equal((await blind.exported()).out.calibration_status,'draft','no image, no reviewed export');
 // A missing camera-check still revokes a tick made on another frame.
 const partial=load(html,{missing:src=>src.endsWith('.jpg')});
 partial.get('checkedCourt').checked=true;partial.get('checkedCourt').onchange();assert.equal(partial.run('draft.calibration_status'),'reviewed');
 partial.get('viewFrame').value=packet.camera_check.samples[0].image;partial.get('viewFrame').onchange();
 assert.equal(partial.run('draft.calibration_status'),'draft');assert.equal(partial.get('checkedCourt').checked,false);
 assert.equal(partial.get('checkedCourt').disabled,true);
 console.log('Calibration UI logic checks passed: carried-draft mode of the current template.');
}

(async()=>{
 await standardDesk(historic,'given built page');
 const base=dataOf(historic);
 await standardDesk(fill(base),'current template, standard desk');
 await draftDesk(base);
 console.log('Calibration UI logic checks passed (fake DOM; no actual browser verification).');
})().catch(e=>{console.error(e);process.exitCode=1;});
