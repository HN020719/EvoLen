'use strict';
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let tokenAnimations = [], stageTimers = [];
function stopTokenMotion() {
 tokenAnimations.forEach(animation => animation.cancel()); tokenAnimations = [];
 stageTimers.forEach(timer => clearTimeout(timer)); stageTimers = [];
}
function animateTokens() {
 stopTokenMotion();
 const stage = document.getElementById('motion-stage');
 if (reducedMotion.matches) { stage.textContent = 'Functional motif preserved as one token'; return; }
 stage.textContent = '1 / Locate the functional motif';
 const play = (element, frames, options) => {
  const animation = element.animate(frames, options); tokenAnimations.push(animation); return animation;
 };
 play(document.querySelector('#sequence span'), [
  {backgroundColor:'#e1edfa',boxShadow:'0 0 0 0 transparent'},
  {backgroundColor:'#b9dcff',boxShadow:'0 0 0 5px #b9dcff44'},
  {backgroundColor:'#e1edfa',boxShadow:'0 0 0 0 transparent'}
 ], {duration:1100,easing:'ease-in-out'});
 document.querySelectorAll('#bpe-tokens > span').forEach((token,i) => {
  play(token,[{opacity:0,transform:'translateY(-12px)'},{opacity:1,transform:'translateY(0)'}],
   {duration:450,delay:850+i*100,fill:'backwards',easing:'cubic-bezier(.2,.7,.2,1)'});
 });
 document.querySelectorAll('#evolen-tokens > span').forEach((token,i) => {
  play(token,[{opacity:0,transform:`translateX(${(1.5-i)*16}px) translateY(8px)`},{opacity:1,transform:'translateX(0) translateY(0)'}],
   {duration:650,delay:2100+i*120,fill:'backwards',easing:'cubic-bezier(.2,.7,.2,1)'});
 });
 play(document.querySelector('#evolen-tokens .intact'),[
  {boxShadow:'0 0 0 0 #0b539400'}, {boxShadow:'0 0 0 5px #0b539425'}, {boxShadow:'0 0 0 0 #0b539400'}
 ],{duration:900,delay:2900,easing:'ease-in-out'});
 stageTimers.push(setTimeout(()=>stage.textContent='2 / BPE fragments the motif',850));
 stageTimers.push(setTimeout(()=>stage.textContent='3 / EvoLen keeps the motif intact',2100));
 stageTimers.push(setTimeout(()=>stage.textContent='Functional motif preserved as one token',3900));
}
document.getElementById('replay-motion').addEventListener('click',animateTokens);
reducedMotion.addEventListener('change',()=>{
 stopTokenMotion();
 document.getAnimations().forEach(animation=>animation.cancel());
 document.getElementById('motion-stage').textContent='Functional motif preserved as one token';
});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopTokenMotion();document.getElementById('motion-stage').textContent='Functional motif preserved as one token';}});

const examples = {
  LMX1B: {motif:'TTAATTAA',bpe:['GC','ATCG','TTAATT','AATGC','AAC'],evolen:['GCATC','G','TTAATTAA','TGCAAC'],fragments:[2,3]},
  'NKX2-5': {motif:'TTGAGTG',bpe:['GC','ATCG','TTG','AGTG','TGCAAC'],evolen:['GCATC','G','TTGAGTG','TGCAAC'],fragments:[2,3]},
  TCF7: {motif:'ATCAAAG',bpe:['GC','ATCG','ATC','AAAGTGC','AAC'],evolen:['GCATC','G','ATCAAAG','TGCAAC'],fragments:[2,3]}
};
document.getElementById('motif').addEventListener('change',event=>{
 const name=event.target.value, example=examples[name];
 const sequence=document.getElementById('sequence'),mark=document.createElement('span');mark.textContent=example.motif;sequence.replaceChildren('GCATCG',mark,'TGCAAC');
 for(const type of ['bpe','evolen']){document.getElementById(type+'-tokens').replaceChildren(...example[type].map((token,index)=>{const span=document.createElement('span');span.textContent=token;if(type==='bpe'&&example.fragments.includes(index))span.className='fragment';if(type==='evolen'&&token===example.motif)span.className='intact';return span;}));}
 const note=document.getElementById('motif-note'),dot=document.createElement('span');dot.className='legend-dot';note.replaceChildren(dot,`${name} motif represented as a single token.`);
 animateTokens();
});
const labels=[['GUE','Yeast'],['GUE','Mouse'],['GUE','Promoter-300'],['GUE','Promoter-core'],['GUE','Splice'],['GUE','TF binding'],['GBM','Human regulatory'],['GBM','Mouse enhancers'],['GBM','Invertebrates'],['NT','Histone marks'],['NT','Enhancers'],['NT','Promoters'],['NT','Splice'],['Multi-SCREEN','Human cCRE'],['snATAC-seq','Cross-species brain']];
const results={100:[.1,4.1,2.7,-1,-4.2,5.7,.5,9.8,-.4,2.3,1.9,2.5,-2.2,1,9.5],200:[2.1,8.1,.6,-.6,.1,4.4,1.3,19.3,1,3.6,.6,1.9,-.9,-.9,33]};
function renderResults(steps){
 const chart=document.getElementById('benchmark-chart');
 const previousWidths=Array.from(chart.querySelectorAll('.bar'),bar=>getComputedStyle(bar).width);
 chart.replaceChildren(...results[steps].map((value,i)=>{
 const row=document.createElement('div');row.className='benchmark-row';
 const label=document.createElement('div');label.className='benchmark-label';const suite=document.createElement('small');suite.textContent=labels[i][0];label.append(suite,labels[i][1]);
 const track=document.createElement('div');track.className='bar-track';track.setAttribute('aria-hidden','true');const bar=document.createElement('div');bar.className='bar '+(value>=0?'positive':'negative');bar.style.width=(Math.abs(value)/35*100)+'%';track.append(bar);
 const number=document.createElement('span');number.className='benchmark-value';number.textContent=(value>0?'+':'')+value.toFixed(1)+'%';number.style.color=value>=0?'var(--blue)':'var(--red)';row.append(label,track,number);return row;
 }));
 if(!reducedMotion.matches && previousWidths.length){
  chart.querySelectorAll('.bar').forEach((bar,i)=>bar.animate(
   [{width:previousWidths[i]},{width:bar.style.width}],
   {duration:650,delay:i*12,easing:'cubic-bezier(.2,.7,.2,1)'}));
 }
 document.getElementById('benchmark-summary').textContent=steps==='100'?'11 of 15 task groups improve at 100K pretraining steps.':'12 of 15 task groups improve at 200K compute-matched pretraining steps.';
 document.querySelectorAll('[data-steps]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.steps===steps)));
}
document.querySelectorAll('[data-steps]').forEach(button=>button.addEventListener('click',()=>renderResults(button.dataset.steps)));renderResults('100');
const dialog=document.getElementById('figure-dialog');
document.querySelectorAll('[data-figure]').forEach(button=>button.addEventListener('click',()=>{const image=document.getElementById('enlarged-figure');image.src=button.dataset.figure;image.alt=button.querySelector('img').alt;document.getElementById('figure-caption').textContent=button.dataset.caption;dialog.showModal();}));
document.getElementById('close-figure').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog){const rect=dialog.getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)dialog.close();}});
document.getElementById('copy-citation').addEventListener('click',async()=>{const citation=document.getElementById('bibtex').textContent,status=document.getElementById('copy-status');try{if(!navigator.clipboard)throw new Error('Clipboard unavailable');await navigator.clipboard.writeText(citation);status.textContent='Citation copied.';}catch{const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Command+C (Mac) or Ctrl+C to copy.';}});

// Play once on arrival. Replay and motif changes let visitors repeat it.
requestAnimationFrame(animateTokens);
