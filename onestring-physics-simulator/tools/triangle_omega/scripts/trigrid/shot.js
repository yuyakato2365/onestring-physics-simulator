const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({args:['--use-gl=swiftshader','--enable-webgl','--ignore-gpu-blocklist']}); const p = await b.newPage({viewport:{width:1200,height:720}});
  const errs=[]; p.on('console', m => { if (m.type()==='error') errs.push(m.text()); }); p.on('pageerror', e => errs.push(String(e)));
  const html = require('fs').readFileSync('bunny_triangle_panels.html','utf8');
  await p.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body>'+html+'</body></html>', {waitUntil:'networkidle'});
  await p.waitForTimeout(1500); await p.screenshot({path:'viewer_shot.png'}); console.log('errors', errs); await b.close();
})();
