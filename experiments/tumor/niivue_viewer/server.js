import http from 'node:http';
import { createReadStream } from 'node:fs';
import { realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const csp = "default-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; font-src 'self' data:; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'";
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.gz': 'application/gzip', '.png': 'image/png' };

export function createLocalServer({ dist = path.join(here, 'dist'), data = path.join(here, 'public/data') } = {}) {
  return http.createServer(async (req, res) => {
    res.setHeader('Content-Security-Policy', csp);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Cache-Control', 'no-store');
    const finish = (status, message) => { res.writeHead(status); res.end(message); };
    if (!/^127\.0\.0\.1(?::\d+)?$/.test(req.headers.host ?? '')) return finish(403, 'Use 127.0.0.1');
    if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) return finish(403, 'Local same-origin requests only');
    if (!['GET', 'HEAD'].includes(req.method)) return finish(405, 'Read-only server');
    try {
      const pathname = decodeURIComponent(req.url.split('?')[0]);
      if (pathname.includes('\\') || pathname.split('/').some(p => p === '..' || p.startsWith('.'))) {
        return finish(403, 'Invalid path');
      }
      const isData = pathname.startsWith('/data/');
      const root = await realpath(isData ? data : dist);
      const relative = isData ? pathname.slice(6) : pathname === '/' ? 'index.html' : pathname.slice(1);
      const target = await realpath(path.join(root, relative));
      if (!target.startsWith(root + path.sep)) return finish(403, 'Invalid path');
      const info = await stat(target);
      if (!info.isFile()) return finish(404, 'Not found');
      res.writeHead(200, { 'Content-Type': mime[path.extname(target)] ?? 'application/octet-stream',
        'Content-Length': info.size });
      if (req.method === 'HEAD') res.end();
      else createReadStream(target).on('error', error => {
        console.error(error);
        res.destroy();
      }).pipe(res);
    } catch (error) {
      if (error instanceof URIError) return finish(400, 'Malformed request path');
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') {
        return finish(404, 'Not found. Run npm run build and export_chemo_niivue.py first.');
      }
      console.error(error);
      finish(500, 'Unable to read local viewer files; see server diagnostics.');
    }
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT ?? 5173);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
  const server = createLocalServer();
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => console.log(`Local Niivue: http://127.0.0.1:${port}`));
}
