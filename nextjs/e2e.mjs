import { chromium } from 'playwright';
const b = await chromium.launch();
const pg = await b.newPage({ viewport: { width: 1500, height: 1050 } });
const log=(...a)=>console.log(...a);
const net = {};
pg.on('response', r => { for (const k of ['aggregate','series','ohlc','forecast']) if (r.url().includes('/analysis/'+k)) net[k]=r.status(); });
async function clickText(t,opts={}){ const el=pg.getByText(t,opts).first(); await el.waitFor({timeout:8000}); await el.click(); }
async function setSel(pred,val){ const ss=pg.locator('select'); const c=await ss.count(); for(let i=0;i<c;i++){ const o=await ss.nth(i).locator('option').allInnerTexts(); if(pred(o)){ await ss.nth(i).selectOption(val); return true; } } return false; }
async function shot(n){ await pg.screenshot({path:`/tmp/e2e_${n}.png`}); log('shot',n); }
try {
  await pg.goto('http://127.0.0.1:3000/files',{waitUntil:'domcontentloaded',timeout:30000});
  await pg.waitForTimeout(2500);
  await clickText('This node'); await pg.waitForTimeout(1000);
  await clickText('data',{exact:true}); await pg.waitForTimeout(900);
  await clickText('files',{exact:true}); await pg.waitForTimeout(900);
  await clickText('fc_demo.parquet'); await pg.waitForTimeout(2200);
  await clickText('Analyze'); await pg.waitForTimeout(1200);
  // PIVOT
  await clickText('pivot',{exact:true}); await pg.waitForTimeout(400);
  await setSel(o=>o[0]==='(none)'&&o.includes('grp'),'grp');
  await setSel(o=>o.includes('value')&&o.includes('ts')&&o[0]!=='(none)'&&o[0]!=='(index)','value');
  await clickText('run'); await pg.waitForTimeout(1800); await shot('an_pivot');
  // SERIES — set value first, then x
  await clickText('series',{exact:true}); await pg.waitForTimeout(400);
  await setSel(o=>o.includes('value')&&o.includes('ts')&&o[0]!=='(none)'&&o[0]!=='(index)','value');
  await setSel(o=>o[0]==='(index)','ts');
  await clickText('run'); await pg.waitForTimeout(1800); await shot('an_series');
  // CANDLES
  await clickText('candles',{exact:true}); await pg.waitForTimeout(400);
  await setSel(o=>o.includes('value')&&o.includes('ts')&&o[0]!=='(none)'&&o[0]!=='(index)','value');
  await setSel(o=>o[0]==='(index)','ts');
  await clickText('run'); await pg.waitForTimeout(1800); await shot('an_candles');
  log('NET:', JSON.stringify(net));
} catch(e){ log('ERR', e.message); await shot('an_err'); }
await b.close();
