import fs from 'fs';
import path from 'path';

const projectRoot = process.cwd();
const srcDir = path.resolve(projectRoot, 'vlab-chatbot/images');
const altSrc = path.resolve(projectRoot, 'images');
const src = fs.existsSync(srcDir) ? srcDir : altSrc;
const dst = path.resolve(projectRoot, 'vlab-chatbot/public/images');

fs.mkdirSync(dst, { recursive: true });

if (!fs.existsSync(src) || !fs.statSync(src).isDirectory()) {
  console.log('No images directory found at', src);
  process.exit(0);
}

const copy = (from: string, to: string) => {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
}

const walk = (dir: string, base = '') => {
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    const rel = path.join(base, name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) walk(full, rel);
    else copy(full, path.join(dst, rel));
  }
};

walk(src);
console.log('Synced images to', dst);

