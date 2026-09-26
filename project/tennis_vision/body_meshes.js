'use strict';
// Optional WebGL body meshes (three.js). Runs after the main replay script and
// plugs in through `meshRenderer`; without three.js or WebGL the page keeps the
// canvas renderer. Geometry provenance, shown to the viewer:
//   observed  - segment between two observed joints (2D pose lifted onto the
//               2.5D camera-facing plane; depth/orientation not measured)
//   completed - limb added by "Complete missing limbs": animation, not evidence
//   template  - limb thickness, torso depth, head size: fixed visual template
(()=>{
 if(typeof THREE==='undefined')return;
 const canvas3d=$('world3d');let renderer;
 try{renderer=new THREE.WebGLRenderer({canvas:canvas3d,antialias:true});}
 catch(error){$('bodyRenderer').value='lines';$('bodyRenderer').disabled=true;$('meshNote').textContent='WebGL is unavailable in this browser, so bodies are shown as pose lines.';return;}
 renderer.setPixelRatio(Math.min(2,window.devicePixelRatio||1));renderer.setSize(world.width,world.height,false);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#101e23');
 // Same pinhole as project(): focal length = 1.1 x canvas height in pixels.
 const cam=new THREE.PerspectiveCamera(2*Math.atan(1/2.2)*180/Math.PI,world.width/world.height,.1,500);
 scene.add(new THREE.HemisphereLight('#e6f2ff','#1d2b26',2.2));
 const sun=new THREE.DirectionalLight('#ffffff',2.4);sun.position.set(6,-14,20);scene.add(sun);
 const v3=a=>new THREE.Vector3(a[0],a[1],a[2]),add=(a,b)=>a.map((v,i)=>v+b[i]);
 // Court: floor, lines and a translucent net at regulation positions.
 scene.add(new THREE.Mesh(new THREE.PlaneGeometry(18,38),new THREE.MeshLambertMaterial({color:'#24433b'})));
 const courtLines=new THREE.LineBasicMaterial({color:'#e4eee5'}),netLines=new THREE.LineBasicMaterial({color:'#9db8ba'}),grid=new THREE.LineBasicMaterial({color:'#315248'});
 const polyline=(points,material,z=0)=>scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(p[0],p[1],p[2]+z))),material));
 for(let x=-8;x<=8;x+=2)polyline([[x,-19,0],[x,19,0]],grid,.002);
 for(let y=-18;y<=18;y+=2)polyline([[-9,y,0],[9,y,0]],grid,.002);
 D.court_lines.forEach(p=>polyline(p,courtLines,.004));D.net_lines.forEach(p=>polyline(p,netLines));
 {const top=D.net_lines.filter(l=>l.every(p=>p[2]>0)).flat().sort((a,b)=>a[0]-b[0]);
  const heightAt=x=>{for(let i=1;i<top.length;i++)if(x<=top[i][0]){const t=(x-top[i-1][0])/((top[i][0]-top[i-1][0])||1);return top[i-1][2]+t*(top[i][2]-top[i-1][2]);}return top.at(-1)[2];};
  const x0=top[0][0],x1=top.at(-1)[0],pos=[];
  for(let i=0;i<24;i++){const a=x0+(x1-x0)*i/24,b=x0+(x1-x0)*(i+1)/24,za=heightAt(a),zb=heightAt(b);pos.push(a,0,0,b,0,0,b,0,zb,a,0,0,b,0,zb,a,0,za);}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  scene.add(new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:'#b7c9cc',transparent:true,opacity:.22,side:THREE.DoubleSide,depthWrite:false})));}
 // Articulated template (metres). Radii are a visual choice, not measurements.
 const LIMBS=[['left_shoulder','left_elbow',.047],['left_elbow','left_wrist',.038],['right_shoulder','right_elbow',.047],['right_elbow','right_wrist',.038],
  ['left_hip','left_knee',.072],['left_knee','left_ankle',.052],['right_hip','right_knee',.072],['right_knee','right_ankle',.052],
  ['left_shoulder','right_shoulder',.055],['left_hip','right_hip',.075]];
 const HEAD=['left_ear','right_ear','left_eye','right_eye','nose'];
 const unitCylinder=new THREE.CylinderGeometry(1,1,1,14,1,true),unitSphere=new THREE.SphereGeometry(1,16,12),unitBox=new THREE.BoxGeometry(1,1,1);
 const translucent=(color,opacity)=>new THREE.MeshLambertMaterial({color,transparent:true,opacity,depthWrite:false});
 const materials={near:new THREE.MeshLambertMaterial({color:'#ffb75b'}),far:new THREE.MeshLambertMaterial({color:'#83c8ff'}),
  completed:translucent('#a9b4b8',.38),placeholder_near:translucent('#ffb75b',.22),placeholder_far:translucent('#83c8ff',.22),predicted:translucent('#a0aaaf',.22)};
 // Mesh pools per player, reused every frame; nothing is allocated while playing.
 const pools={};let used={};
 function take(id,geometry,material){const key=id+geometry.uuid,list=pools[key]||(pools[key]=[]),n=used[key]||0;used[key]=n+1;
  if(!list[n]){list[n]=new THREE.Mesh(geometry,material);scene.add(list[n]);}const mesh=list[n];mesh.material=material;mesh.visible=true;mesh.quaternion.identity();return mesh;}
 const Y=new THREE.Vector3(0,1,0);
 function capsule(id,a,b,r,material){const A=v3(a),B=v3(b),dir=B.clone().sub(A),len=dir.length();if(len<1e-4)return;
  const c=take(id,unitCylinder,material);c.position.copy(A).add(B).multiplyScalar(.5);c.quaternion.setFromUnitVectors(Y,dir.normalize());c.scale.set(r,len,r);
  for(const P of [A,B]){const s=take(id,unitSphere,material);s.position.copy(P);s.scale.set(r,r,r);}}
 function sphere(id,p,r,material,squash=1){const s=take(id,unitSphere,material);s.position.copy(v3(p));s.scale.set(r,r,r*squash);}
 const mean=points=>points.reduce((s,q)=>s.map((v,i)=>v+q[i]/points.length),[0,0,0]);
 function torso(id,shoulders,hips,material){const top=v3(mean(shoulders)),bottom=v3(mean(hips)),up=top.clone().sub(bottom),len=up.length();if(len<.15)return false;up.normalize();
  const across=(shoulders.length===2?v3(shoulders[1]).sub(v3(shoulders[0])):hips.length===2?v3(hips[1]).sub(v3(hips[0])):new THREE.Vector3(1,0,0));
  across.sub(up.clone().multiplyScalar(across.dot(up)));if(across.lengthSq()<1e-6)across.set(1,0,0);across.normalize();
  const width=Math.min(.46,Math.max(.26,shoulders.length===2?v3(shoulders[0]).distanceTo(v3(shoulders[1])):.36));
  const box=take(id,unitBox,material);box.position.copy(top).add(bottom).multiplyScalar(.5);
  box.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(across,up,new THREE.Vector3().crossVectors(across,up).normalize()));box.scale.set(width,len*.92,.2);return true;}
 // Animation completion for limbs whose joints were not observed: rest pose
 // hanging from the nearest observed joint. Never evidence; drawn grey.
 function completeLimbs(j){const out={},down=(p,d)=>[p[0],p[1],Math.max(0,p[2]-d)];
  for(const side of ['left','right']){const S=j[side+'_shoulder'],H=j[side+'_hip'];
   if(S&&!j[side+'_elbow'])out[side+'_elbow']=down(S,.29);
   const E=j[side+'_elbow']||out[side+'_elbow'];if(E&&!j[side+'_wrist'])out[side+'_wrist']=down(E,.26);
   if(H&&!j[side+'_knee'])out[side+'_knee']=[H[0],H[1],H[2]*.5];
   const K=j[side+'_knee']||out[side+'_knee'];if(K&&!j[side+'_ankle'])out[side+'_ankle']=[K[0],K[1],0];}
  return out;}
 function drawPlayer(p,delta){const id=p.identity_id,counts={observed:0,completed:0};
  if(p.predicted||p.avatar!=='pose_wireframe'){const feet=$('smooth').checked?p.feet_xyz_m:p.raw_feet_xyz_m;
   capsule(id,[feet[0],feet[1],.25],[feet[0],feet[1],1.55],.24,p.predicted?materials.predicted:materials['placeholder_'+id]);return {placeholder:true,...counts};}
  const j={};for(const [name,point] of Object.entries(p.joints_m))j[name]=add(point,delta);
  const extra=$('completeLimbs').checked?completeLimbs(j):{},solid=materials[id];
  for(const [a,b,r] of LIMBS){const A=j[a]||extra[a],B=j[b]||extra[b];if(!A||!B)continue;const observed=!!(j[a]&&j[b]);capsule(id,A,B,r,observed?solid:materials.completed);counts[observed?'observed':'completed']++;}
  const shoulders=['left_shoulder','right_shoulder'].filter(k=>j[k]).map(k=>j[k]),hips=['left_hip','right_hip'].filter(k=>j[k]).map(k=>j[k]);
  if(shoulders.length&&hips.length&&torso(id,shoulders,hips,solid))counts.observed++;
  const headPoints=HEAD.filter(k=>j[k]).map(k=>j[k]);
  if(headPoints.length){const c=mean(headPoints);sphere(id,c,.105,solid);counts.observed++;if(shoulders.length)capsule(id,mean(shoulders),[c[0],c[1],c[2]-.08],.05,solid);}
  for(const side of ['left','right']){const W=j[side+'_wrist']||extra[side+'_wrist'],A=j[side+'_ankle']||extra[side+'_ankle'];
   if(W)sphere(id,W,.045,j[side+'_wrist']?solid:materials.completed);
   if(A)sphere(id,[A[0],A[1],Math.max(.035,A[2])],.075,j[side+'_ankle']?solid:materials.completed,.45);}
  return {placeholder:false,...counts};}
 // Racquets: grip at the supporting wrist(s), axis toward the detected box centre.
 // Handle/throat/head proportions and the face plane are display choices.
 const unitRing=new THREE.TorusGeometry(1,.09,6,28),unitDisc=new THREE.CircleGeometry(1,24);
 const racquetMaterial=(colour,opacity)=>new THREE.MeshLambertMaterial({color:colour,transparent:opacity<1,opacity,depthWrite:opacity>=1,side:THREE.DoubleSide});
 const racquetMaterials={};
 function racquetMat(kind,opacity){const key=kind+Math.round(opacity*10);return racquetMaterials[key]||(racquetMaterials[key]=racquetMaterial(kind==='strings'?'#cfdadd':kind==='stale'?'#a9b4b8':'#f3efe2',kind==='strings'?opacity*.35:opacity));}
 function drawRacquet(r,delta){const grip=v3(add(r.grip_m,delta)),axis=v3(r.axis_unit).normalize(),L=r.length_m,stale=r.status==='stale';
  const opacity=stale?.3:Math.max(.35,Math.min(1,r.confidence*1.6)),frame=racquetMat(stale?'stale':'frame',opacity),id='racquet-'+r.identity_id;
  const butt=grip.clone().addScaledVector(axis,-.06),throat=grip.clone().addScaledVector(axis,.30);capsule(id,butt.toArray(),throat.toArray(),.016,frame);
  // Face plane: contains the axis and faces the camera as far as possible (display choice, not measured).
  const toCam=v3(view.eye).sub(grip).normalize(),normal=toCam.sub(axis.clone().multiplyScalar(toCam.dot(axis)));if(normal.lengthSq()<1e-6)normal.set(0,0,1);normal.normalize();
  const width=new THREE.Vector3().crossVectors(axis,normal).normalize(),centre=grip.clone().addScaledVector(axis,L-.165);
  const basis=new THREE.Matrix4().makeBasis(width,axis,normal);
  const ring=take(id,unitRing,frame);ring.position.copy(centre);ring.quaternion.setFromRotationMatrix(basis);ring.scale.set(.125,.165,.125);
  const strings=take(id,unitDisc,racquetMat('strings',opacity));strings.position.copy(centre);strings.quaternion.setFromRotationMatrix(basis);strings.scale.set(.12,.16,1);
  for(const side of [-1,1]){const shoulder=centre.clone().addScaledVector(axis,-.15).addScaledVector(width,side*.055);capsule(id,throat.toArray(),shoulder.toArray(),.012,frame);}}
 function render(state){
  canvas3d.hidden=false;world.classList.add('overMesh');used={};
  for(const list of Object.values(pools))for(const m of list)m.visible=false;
  const notes=[];
  for(const p of state.players){const delta=sub($('smooth').checked?p.feet_xyz_m:p.raw_feet_xyz_m,p.feet_xyz_m),r=drawPlayer(p,delta);
   let text=`${p.identity_id}: ${r.placeholder?(p.predicted?'predicted position, no pose (grey placeholder)':'no usable pose (translucent placeholder)'):`${r.observed} observed segments${r.completed?`, ${r.completed} completed (animation)`:''}`}`;
   const racquet=(state.racquets||[]).find(q=>q.identity_id===p.identity_id);
   if(racquet&&$('showRacquets').checked&&!r.placeholder){drawRacquet(racquet,delta);
    text+=` · racquet ${racquet.status==='stale'?`held ${racquet.age_frames} frame(s) after last box (stale)`:`box observed (conf ${racquet.confidence.toFixed(2)})`}, ${racquet.hand==='both'?'both hands':racquet.hand+' hand'}, orientation estimated`;}
   notes.push(text);}
  $('meshNote').textContent=`3D meshes · ${notes.join(' / ')||'no players at this frame'} · depth, orientation and limb thickness are not measured.`;
  cam.position.set(...view.eye);cam.up.set(...view.up);cam.lookAt(...add(view.eye,view.forward));
  renderer.render(scene,cam);}
 function off(){canvas3d.hidden=true;world.classList.remove('overMesh');$('meshNote').textContent='';}
 meshRenderer={enabled(){const on=$('bodyRenderer').value==='mesh';if(!on)off();return on;},render,scene,materials};
 $('bodyRenderer').onchange=drawWorld;
 $('meshWireframe').onchange=()=>{for(const m of Object.values(materials))m.wireframe=$('meshWireframe').checked;drawWorld();};
 $('completeLimbs').onchange=drawWorld;$('showRacquets').onchange=drawWorld;
 drawWorld();
})();
