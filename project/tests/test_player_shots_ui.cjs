// Offline logic checks. Does not claim browser rendering or video decoding QA.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
const payload=JSON.parse(scripts.find(s=>s[1].includes('application/json'))[2]);
class Element{
 constructor(id){this.id=id;this.children=[];this.handlers={};this.value='';this.textContent='';this.width=480;this.height=360;this.currentTime=0;this.readyState=2;this.paused=true;this.checked=true;this.hidden=false;this.draws=[];}
 append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}
 addEventListener(t,h){this.handlers[t]=h;}click(){return this.onclick?.();}
 getContext(){return new Proxy({},{get:(o,k)=>o[k]??((...args)=>this.draws.push({fn:k,args})),set:(o,k,v)=>(o[k]=v,true)});}
 pause(){this.paused=true;}play(){this.paused=false;return Promise.resolve();}setPointerCapture(){}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
get('replay-data').textContent=JSON.stringify(payload);let download;
const context=vm.createContext({document:{getElementById:get,createElement:n=>new Element(n)},localStorage:{getItem:()=>null,setItem(){}},Blob,
 URL:{createObjectURL:b=>(download=b,'fake'),revokeObjectURL(){}},setTimeout:f=>f(),requestAnimationFrame(){},console});
vm.runInContext(scripts.find(s=>!s[1].includes('application/json'))[2],context);
const run=s=>vm.runInContext(s,context),statuses=JSON.stringify(payload.events.map(e=>e.status));
assert.ok(run("kinds.includes('overhead')"));
assert.equal(get('shotList').children.length,payload.shots.shots.filter(s=>!s.action_state||['shot_candidate','confirmed_shot'].includes(s.action_state)).length);
get('activityFilter').value='all';get('activityFilter').onchange();
assert.equal(get('shotList').children.length,payload.shots.shots.length);
const index=payload.shots.shots.findIndex(s=>s.contact_candidate);
assert.ok(index>=0,'real candidate available');get('shotList').children[index].click();
assert.match(get('shotSummary').textContent,/Contact support:/);
assert.equal(run('frame'),payload.shots.shots[index].contact_candidate.frame);
assert.ok(get('imageOverlay').draws.some(d=>d.fn==='fillText'&&String(d.args[0]).includes('contact?')));
assert.ok(get('farZoom').draws.some(d=>d.fn==='drawImage'));
assert.match(get('farQuality').textContent,/supported joints/);
if(payload.report.contact_review_binding){
 get('video').seeking=true;get('confirmContact').click();assert.equal(run('Object.keys(contactLabels).length'),0);
 get('video').seeking=false;get('contactPlayer').value='far';get('confirmContact').click();
 assert.equal(run('contactLabels[selected.event_id].status'),'confirmed');
 assert.equal(run('contactLabels[selected.event_id].player_id'),'far');
 get('unsureContact').click();assert.equal(run('contactLabels[selected.event_id].status'),'uncertain');
 get('exportContacts').click();const contactDownload=download;
 contactDownload.text().then(text=>{const data=JSON.parse(text);assert.equal(data.binding,payload.report.contact_review_binding);assert.equal(data.labels[0].status,'uncertain');});
}
run('D.frames[frame].players=[];drawFarZoom()');assert.match(get('farQuality').textContent,/No observed/);
get('shotType').value='forehand';get('strokeHands').value='both';get('saveType').click();get('export').click();
(async()=>{const correction=JSON.parse(await download.text()).corrections[0];assert.equal(correction.shot_type,'forehand');assert.equal(correction.striking_hands,'both');
 assert.equal(run('JSON.stringify(D.events.map(e=>e.status))'),statuses);console.log('Player/contacts replay UI logic checks passed');})();
