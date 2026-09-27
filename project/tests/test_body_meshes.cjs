// Logic checks for tennis_vision/body_meshes.js with a minimal fake THREE and fake DOM.
// Not a WebGL or visual test: it checks which meshes/materials the script asks for.
// Usage: node tests/test_body_meshes.cjs [replay-data.json]
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const root=path.join(__dirname,'..');
const template=fs.readFileSync(path.join(root,'tennis_vision/shot_replay.html'),'utf8');
const meshCode=fs.readFileSync(path.join(root,'tennis_vision/body_meshes.js'),'utf8');
const payload=JSON.parse(fs.readFileSync(process.argv[2]||path.join(root,'runs/claude-baseline-racquets/replay-data.json'),'utf8'));
const scripts=[...template.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const mainCode=scripts.find(s=>!s[1].includes('application/json')&&!s[1].includes('id='))[2];
// The page loads three.js, then body_meshes.js, after the main replay script.
const order=scripts.map(s=>(s[1].match(/id="([^"]+)"/)||[])[1]||'main');
assert.deepEqual(order.slice(-3),['main','three-vendor','body-meshes']);
assert.equal(scripts.at(-1)[2],'__BODY_MESHES_JS__');

// ---- Minimal fake THREE: real vector maths, bookkeeping for everything else. ----
let uuid=0;
class Vector3{
 constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z;}
 set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}copy(v){return this.set(v.x,v.y,v.z);}clone(){return new Vector3(this.x,this.y,this.z);}
 add(v){this.x+=v.x;this.y+=v.y;this.z+=v.z;return this;}sub(v){this.x-=v.x;this.y-=v.y;this.z-=v.z;return this;}
 multiplyScalar(s){this.x*=s;this.y*=s;this.z*=s;return this;}addScaledVector(v,s){return this.add(v.clone().multiplyScalar(s));}
 dot(v){return this.x*v.x+this.y*v.y+this.z*v.z;}lengthSq(){return this.dot(this);}length(){return Math.sqrt(this.lengthSq());}
 normalize(){return this.multiplyScalar(1/(this.length()||1));}distanceTo(v){return this.clone().sub(v).length();}
 crossVectors(a,b){return this.set(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}toArray(){return [this.x,this.y,this.z];}
}
class Quaternion{identity(){this.basis=null;return this;}setFromUnitVectors(a,b){this.basis=[a.clone(),b.clone()];return this;}setFromRotationMatrix(m){this.basis=m;return this;}}
class Object3D{constructor(){this.position=new Vector3();this.scale=new Vector3(1,1,1);this.quaternion=new Quaternion();this.up=new Vector3(0,1,0);this.visible=true;this.children=[];}
 add(o){this.children.push(o);return this;}lookAt(...a){this.target=a;}}
const geometry=type=>class{constructor(...args){this.type=type;this.args=args;this.uuid='g'+(++uuid);}setFromPoints(p){this.points=p;return this;}setAttribute(k,v){this[k]=v;return this;}};
const material=type=>class{constructor(o={}){Object.assign(this,{type,transparent:false,opacity:1,wireframe:false},o);}};
const renders=[];let webglFails=false;
const THREE={Vector3,Scene:Object3D,
 Color:class{constructor(c){this.hex=c;}},
 WebGLRenderer:class{constructor(o){if(webglFails)throw new Error('no WebGL');this.options=o;}setPixelRatio(){}setSize(...a){this.size=a;}render(scene,cam){renders.push({scene,cam});}},
 PerspectiveCamera:class extends Object3D{constructor(...a){super();this.args=a;}},
 HemisphereLight:class extends Object3D{},DirectionalLight:class extends Object3D{},
 Mesh:class extends Object3D{constructor(g,m){super();this.isMesh=true;this.geometry=g;this.material=m;}},
 Line:class extends Object3D{constructor(g,m){super();this.geometry=g;this.material=m;}},
 Matrix4:class{makeBasis(...a){this.basis=a;return this;}},
 BufferGeometry:geometry('BufferGeometry'),Float32BufferAttribute:class{constructor(a,n){this.array=a;this.itemSize=n;}},
 PlaneGeometry:geometry('PlaneGeometry'),CylinderGeometry:geometry('CylinderGeometry'),SphereGeometry:geometry('SphereGeometry'),
 BoxGeometry:geometry('BoxGeometry'),TorusGeometry:geometry('TorusGeometry'),CircleGeometry:geometry('CircleGeometry'),
 MeshLambertMaterial:material('MeshLambertMaterial'),MeshBasicMaterial:material('MeshBasicMaterial'),LineBasicMaterial:material('LineBasicMaterial'),
 DoubleSide:2};

// ---- Fake DOM with the template's own control defaults. ----
class Element{
 constructor(id){this.id=id;this.children=[];this.handlers={};this.value='';this.textContent='';this.width=1000;this.height=562;this.currentTime=0;this.readyState=1;this.paused=true;this.checked=false;this.hidden=false;this.disabled=false;this.draws=[];this.attributes={};
  const classes=new Set();this.classList={add:c=>classes.add(c),remove:c=>classes.delete(c),contains:c=>classes.has(c),toggle:(c,on)=>((on??!classes.has(c))?classes.add(c):classes.delete(c))};}
 get value(){return this._value;}set value(v){this._value=String(v);}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}
 addEventListener(type,handler){this.handlers[type]=handler;}removeEventListener(type){delete this.handlers[type];}click(){return this.onclick?.();}
 setAttribute(name,value){this.attributes[name]=String(value);}getAttribute(name){return this.attributes[name]??null;}removeAttribute(name){delete this.attributes[name];}
 getContext(){return new Proxy({},{get:(o,k)=>k in o?o[k]:((...args)=>this.draws.push({fn:k,args,stroke:o.strokeStyle})),set:(o,k,v)=>(o[k]=v,true)});}
 pause(){this.paused=true;this.handlers.pause?.();}play(){this.paused=false;this.handlers.play?.();return Promise.resolve();}setPointerCapture(){}
}
function load({failWebGL=false}={}){
 webglFails=failWebGL;renders.length=0;
 const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
 for(const m of template.matchAll(/<input id="(\w+)" type="checkbox"( checked)?>/g))get(m[1]).checked=!!m[2];
 for(const m of template.matchAll(/<select id="(\w+)">([\s\S]*?)<\/select>/g)){const options=[...m[2].matchAll(/<option value="([^"]*)"( selected)?/g)];get(m[1]).value=(options.find(o=>o[2])||options[0])[1];}
 get('replay-data').textContent=JSON.stringify(payload);
 // Page-level listeners (e.g. keyboard shortcuts) are recorded, not dispatched, unless a check calls them.
 const listeners={document:{},window:{}},target=kind=>({addEventListener:(type,handler)=>{(listeners[kind][type]??=[]).push(handler);},removeEventListener:(type,handler)=>{listeners[kind][type]=(listeners[kind][type]||[]).filter(h=>h!==handler);}});
 const context=vm.createContext({document:{getElementById:get,createElement:name=>new Element(name),...target('document')},window:{devicePixelRatio:1,...target('window')},
  localStorage:{getItem:()=>null,setItem(){}},Blob,URL:{createObjectURL:()=>'fake',revokeObjectURL(){}},setTimeout:f=>f(),requestAnimationFrame(){},console});
 vm.runInContext(mainCode,context);
 const run=s=>vm.runInContext(s,context);
 assert.equal(run('meshRenderer'),null,'canvas renderer until the mesh script runs');
 context.THREE=THREE;vm.runInContext(meshCode,context);
 return {get,run,context,listeners};
}

// ---- Independent expectations (the documented rules, not the implementation). ----
const LIMBS=[['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],
 ['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle'],['left_shoulder','right_shoulder'],['left_hip','right_hip']];
const HEAD=['left_ear','right_ear','left_eye','right_eye','nose'];
const mean=ps=>[0,1,2].map(i=>ps.reduce((s,p)=>s+p[i],0)/ps.length);
function completable(j){const out=new Set();
 for(const s of ['left','right']){if(j[s+'_shoulder']&&!j[s+'_elbow'])out.add(s+'_elbow');if((j[s+'_elbow']||out.has(s+'_elbow'))&&!j[s+'_wrist'])out.add(s+'_wrist');
  if(j[s+'_hip']&&!j[s+'_knee'])out.add(s+'_knee');if((j[s+'_knee']||out.has(s+'_knee'))&&!j[s+'_ankle'])out.add(s+'_ankle');}
 return out;}
function expected(p,complete){
 if(p.predicted||p.avatar!=='pose_wireframe')return {placeholder:true};
 const j=p.joints_m,extra=complete?completable(j):new Set(),has=k=>!!j[k]||extra.has(k);let observed=0,completed=0;
 for(const [a,b] of LIMBS){if(j[a]&&j[b])observed++;else if(has(a)&&has(b))completed++;}
 const shoulders=['left_shoulder','right_shoulder'].filter(k=>j[k]).map(k=>j[k]),hips=['left_hip','right_hip'].filter(k=>j[k]).map(k=>j[k]);
 if(shoulders.length&&hips.length&&Math.hypot(...mean(shoulders).map((v,i)=>v-mean(hips)[i]))>=.15)observed++;
 if(HEAD.some(k=>j[k]))observed++;
 return {placeholder:false,observed,completed};}
function parseNote(text){const out={};
 for(const part of text.replace(/^3D meshes · /,'').split(' / ')){const m=part.match(/^(\w+): (.*)$/);if(!m)continue;const body=m[2];
  const counts=body.match(/^(\d+) observed segments(?:, (\d+) completed \(animation\))?/);
  out[m[1]]={placeholder:/placeholder/.test(body),predicted:/^predicted position, no pose/.test(body),observed:counts?+counts[1]:null,completed:counts?+(counts[2]||0):null,racquet:/racquet/.test(body),stale:/\(stale\)/.test(body)};}
 return out;}

const {get,run}=load();
const M=run('meshRenderer'),scene=M.scene,materials=M.materials;
// Court floor (plane) and net (buffer geometry) are static; bodies and racquets use the unit primitives.
const POOLED=['CylinderGeometry','SphereGeometry','BoxGeometry','TorusGeometry','CircleGeometry'];
const bodyMeshes=()=>scene.children.filter(o=>o.isMesh&&o.visible&&POOLED.includes(o.geometry.type));
const renderAlone=(player,racquets=[])=>{M.render({players:[player],racquets});return bodyMeshes();};
const frames=payload.frames.map((f,i)=>[i,f]).filter(([,f])=>f&&f.players);
const players=frames.flatMap(([i,f])=>f.players.map(p=>({i,f,p})));
assert.ok(M&&typeof M.render==='function'&&typeof M.enabled==='function');
// Default page state: meshes on, WebGL canvas shown, completion and wireframe off.
assert.equal(get('bodyRenderer').value,'mesh');assert.equal(get('world3d').hidden,false);assert.ok(get('world').classList.contains('overMesh'));
assert.equal(get('completeLimbs').checked,false,'Complete missing limbs is off by default');
assert.equal(get('meshWireframe').checked,false);assert.equal(get('showRacquets').checked,true);
assert.ok(renders.length>0,'mesh script draws once on load');

// 1. Observed vs completed counts on every real frame, completion off by default.
let placeholders=0,partial=0;
for(const [i,f] of frames){
 run(`frame=${i};drawWorld()`);const notes=parseNote(get('meshNote').textContent);
 for(const p of f.players){const want=expected(p,false),got=notes[p.identity_id];assert.ok(got,`frame ${i} ${p.identity_id} has a note`);
  assert.equal(got.placeholder,want.placeholder,`frame ${i} ${p.identity_id} placeholder`);
  if(want.placeholder){placeholders++;continue;}
  if(want.observed<12)partial++;
  assert.equal(got.observed,want.observed,`frame ${i} ${p.identity_id} observed`);assert.equal(got.completed,0,`frame ${i} completion off by default`);}
 assert.ok(!bodyMeshes().some(m=>m.material===materials.completed),`frame ${i}: no completed (grey) limbs by default`);}
assert.ok(placeholders>0,'real data has a player without usable pose');assert.ok(partial>0,'real data has partial poses');

// 2. Placeholder for players with no pose: one translucent capsule, never a limb or racquet.
const noPose=players.find(x=>x.p.avatar!=='pose_wireframe'&&!x.p.predicted);assert.ok(noPose,'real no-pose player exists');
{const p=noPose.p,fake={identity_id:p.identity_id,status:'observed',hand:'right',age_frames:0,confidence:.9,grip_m:[...p.feet_xyz_m.slice(0,2),1],axis_unit:[0,0,1],length_m:.685};
 const meshes=renderAlone(p,[fake]);// synthetic racquet for this player: must still not be drawn
 assert.deepEqual(meshes.map(m=>m.geometry.type).sort(),['CylinderGeometry','SphereGeometry','SphereGeometry']);
 const mat=materials['placeholder_'+p.identity_id];assert.ok(mat,'placeholder material exists');assert.ok(meshes.every(m=>m.material===mat));
 assert.ok(mat.transparent&&mat.opacity<.5&&mat.depthWrite===false,'placeholder is translucent');assert.notEqual(mat,materials[p.identity_id]);
 assert.ok(!meshes.some(m=>['TorusGeometry','CircleGeometry'].includes(m.geometry.type)),'no racquet on placeholders');
 const note=parseNote(get('meshNote').textContent)[p.identity_id];assert.ok(note.placeholder&&!note.racquet,'placeholder note has no racquet');}
{const p=structuredClone(players.find(x=>x.p.avatar==='pose_wireframe').p);p.predicted=true;
 const meshes=renderAlone(p,[{identity_id:p.identity_id,status:'observed',hand:'left',age_frames:0,confidence:.9,grip_m:[0,0,1],axis_unit:[0,0,1],length_m:.685}]);
 assert.equal(meshes.length,3);assert.ok(meshes.every(m=>m.material===materials.predicted),'predicted players are a grey placeholder even with joints');
 const note=parseNote(get('meshNote').textContent)[p.identity_id];assert.ok(note.predicted&&note.placeholder&&!note.racquet);}

// 3. Completion adds grey limbs only where joints are missing; observed limbs keep the player colour.
const grey=materials.completed;assert.equal(grey.color,'#a9b4b8');assert.ok(grey.transparent&&grey.opacity<1);
assert.notEqual(grey,materials.near);assert.notEqual(grey,materials.far);
const incomplete=players.find(x=>{const e=expected(x.p,true);return !e.placeholder&&e.completed>0;});assert.ok(incomplete,'real frame with missing limb joints');
get('completeLimbs').checked=true;get('completeLimbs').onchange();
{run(`frame=${incomplete.i};drawWorld()`);const note=parseNote(get('meshNote').textContent)[incomplete.p.identity_id],want=expected(incomplete.p,true);
 assert.equal(note.observed,want.observed);assert.equal(note.completed,want.completed);assert.match(get('meshNote').textContent,/completed \(animation\)/);
 const meshes=renderAlone(incomplete.p),cylinders=meshes.filter(m=>m.geometry.type==='CylinderGeometry');
 assert.equal(cylinders.filter(m=>m.material===grey).length,want.completed,'one grey capsule per completed limb');
 const solid=materials[incomplete.p.identity_id];
 // observed limbs (+ neck when a head is seen) stay solid; the torso is a box, the head a sphere
 const neck=HEAD.some(k=>incomplete.p.joints_m[k])&&(incomplete.p.joints_m.left_shoulder||incomplete.p.joints_m.right_shoulder)?1:0;
 assert.equal(cylinders.filter(m=>m.material===solid).length,LIMBS.filter(([a,b])=>incomplete.p.joints_m[a]&&incomplete.p.joints_m[b]).length+neck);
 assert.ok(meshes.every(m=>m.material===solid||m.material===grey),'completion never uses another colour');
 const missingWrist=['left','right'].find(s=>!incomplete.p.joints_m[s+'_wrist']);
 if(missingWrist)assert.ok(meshes.some(m=>m.geometry.type==='SphereGeometry'&&m.material===grey&&m.scale.x===.045),'completed wrist is grey');}
// Every frame: completed counts follow the documented rest-pose rules.
for(const [i,f] of frames){run(`frame=${i};drawWorld()`);const notes=parseNote(get('meshNote').textContent);
 for(const p of f.players){const want=expected(p,true);if(want.placeholder)continue;assert.equal(notes[p.identity_id].observed,want.observed);assert.equal(notes[p.identity_id].completed,want.completed,`frame ${i} completed`);}}
get('completeLimbs').checked=false;get('completeLimbs').onchange();
assert.ok(!bodyMeshes().some(m=>m.material===grey),'turning completion off removes grey limbs');

// 4. Racquets: observed frame is off-white, stale is grey; toggle hides them.
const racquetMeshes=()=>bodyMeshes().filter(m=>m.geometry.type==='TorusGeometry');
const withRacquet=status=>frames.find(([,f])=>(f.racquets||[]).some(r=>r.status===status&&f.players.some(p=>p.identity_id===r.identity_id&&p.avatar==='pose_wireframe'&&!p.predicted)));
for(const status of ['observed','stale']){const hit=withRacquet(status);assert.ok(hit,`real frame with a ${status} racquet`);
 const [i,f]=hit,r=f.racquets.find(q=>q.status===status),p=f.players.find(q=>q.identity_id===r.identity_id);
 renderAlone(p,[r]);const rings=racquetMeshes();assert.equal(rings.length,1);
 const frame=rings[0].material;
 if(status==='stale'){assert.equal(frame.color,'#a9b4b8','stale racquet is grey');assert.ok(frame.transparent&&Math.abs(frame.opacity-.3)<1e-9);}
 else{assert.equal(frame.color,'#f3efe2');assert.notEqual(frame.color,'#a9b4b8');}
 assert.match(get('meshNote').textContent,status==='stale'?/\(stale\).*orientation estimated/:/box observed \(conf \d\.\d\d\).*orientation estimated/);
 run(`frame=${i};drawWorld()`);assert.ok(parseNote(get('meshNote').textContent)[p.identity_id].racquet);
 get('showRacquets').checked=false;get('showRacquets').onchange();
 assert.equal(racquetMeshes().length,0,'racquets hidden by the toggle');assert.ok(!parseNote(get('meshNote').textContent)[p.identity_id].racquet);
 get('showRacquets').checked=true;get('showRacquets').onchange();}
// No real frame draws a racquet for a placeholder player.
for(const [i,f] of frames){if(!f.players.some(p=>p.avatar!=='pose_wireframe'||p.predicted))continue;run(`frame=${i};drawWorld()`);const notes=parseNote(get('meshNote').textContent);
 for(const p of f.players)if(notes[p.identity_id].placeholder)assert.ok(!notes[p.identity_id].racquet,`frame ${i}: no racquet on placeholder`);}

// 5. Wireframe toggles every body material (and nothing else is left solid).
run(`frame=${incomplete.i};drawWorld()`);
get('meshWireframe').checked=true;get('meshWireframe').onchange();
for(const [name,m] of Object.entries(materials))assert.equal(m.wireframe,true,`${name} wireframe on`);
assert.ok(bodyMeshes().filter(m=>!['TorusGeometry','CircleGeometry'].includes(m.geometry.type)).every(m=>m.material.wireframe));
get('meshWireframe').checked=false;get('meshWireframe').onchange();
for(const [name,m] of Object.entries(materials))assert.equal(m.wireframe,false,`${name} wireframe off`);
assert.deepEqual(Object.keys(materials).sort(),['completed','far','near','placeholder_far','placeholder_near','predicted']);

// 6. Pose lines mode hides the WebGL canvas and falls back to canvas pose lines.
const before=renders.length;get('world').draws=[];
get('bodyRenderer').value='lines';get('bodyRenderer').onchange();
assert.equal(get('world3d').hidden,true);assert.ok(!get('world').classList.contains('overMesh'));assert.equal(get('meshNote').textContent,'');
assert.equal(renders.length,before,'no WebGL render in Pose lines mode');
assert.ok(get('world').draws.some(d=>d.fn==='lineTo'),'canvas draws pose lines instead');
get('bodyRenderer').value='mesh';get('bodyRenderer').onchange();
assert.equal(get('world3d').hidden,false);assert.ok(get('world').classList.contains('overMesh'));assert.equal(renders.length,before+1);

// 7. Without WebGL the page keeps canvas pose lines and says so.
{const fallback=load({failWebGL:true});
 assert.equal(fallback.run('meshRenderer'),null);assert.equal(fallback.get('bodyRenderer').value,'lines');assert.equal(fallback.get('bodyRenderer').disabled,true);
 assert.match(fallback.get('meshNote').textContent,/WebGL is unavailable/);}

console.log(`Body mesh checks passed on ${frames.length} frames (${placeholders} placeholder players; fake THREE + fake DOM, not a WebGL visual test).`);
