#!/usr/bin/env node
/**
 * claude-voice npm wrapper
 *
 * Runs the claude-voice Python MCP server.
 * Requires Python 3.12+ and pip.
 *
 * On first run, installs the Python package into a local venv.
 * Subsequent runs skip installation and start instantly.
 */

const { spawnSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const PKG_NAME = 'claude-voice';
const PKG_VERSION = require('../package.json').version;
const VENV_DIR = path.join(os.homedir(), '.claude-voice', 'venv');
const MARKER = path.join(os.homedir(), '.claude-voice', `installed-${PKG_VERSION}`);

function findPython() {
  for (const bin of ['python3.12', 'python3', 'python']) {
    const result = spawnSync(bin, ['--version'], { encoding: 'utf8' });
    if (result.status === 0) {
      const match = result.stdout.match(/Python (\d+)\.(\d+)/);
      if (match && (parseInt(match[1]) > 3 || (parseInt(match[1]) === 3 && parseInt(match[2]) >= 12))) {
        return bin;
      }
    }
  }
  return null;
}

function run() {
  const python = findPython();
  if (!python) {
    console.error('[claude-voice] ERROR: Python 3.12+ is required.');
    console.error('  Install via: brew install python@3.12');
    process.exit(1);
  }

  // Create venv if it doesn't exist
  if (!fs.existsSync(VENV_DIR)) {
    fs.mkdirSync(path.dirname(VENV_DIR), { recursive: true });
    process.stderr.write('[claude-voice] Creating virtual environment...\n');
    const r = spawnSync(python, ['-m', 'venv', VENV_DIR], { stdio: 'inherit' });
    if (r.status !== 0) { process.exit(r.status); }
  }

  const venvPython = path.join(VENV_DIR, 'bin', 'python');
  const venvPip = path.join(VENV_DIR, 'bin', 'pip');

  // Install or upgrade if not already at this version
  if (!fs.existsSync(MARKER)) {
    // Install from the bundled Python source (no PyPI needed)
    const pkgRoot = path.join(__dirname, '..');
    process.stderr.write(`[claude-voice] Installing Python dependencies (first run, ~500MB download)...\n`);
    const extras = process.platform === 'darwin' ? '.[macos]' : '.';
    const r = spawnSync(venvPip, ['install', extras], { cwd: pkgRoot, stdio: 'inherit' });
    if (r.status !== 0) { process.exit(r.status); }
    fs.writeFileSync(MARKER, PKG_VERSION);
  }

  // Launch the MCP server — inherit stdio so Claude Code can communicate
  const child = spawn(venvPython, ['-m', 'lazy_claude'], { stdio: 'inherit' });
  child.on('exit', (code) => process.exit(code ?? 0));
}

run();
