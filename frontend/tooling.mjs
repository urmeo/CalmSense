#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { delimiter, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import packageConfig from './config/package.mjs';
import packageLock from './config/package-lock.mjs';
import typescriptConfig from './config/typescript.mjs';

const root = dirname(fileURLToPath(import.meta.url));
const nodeDirectory = dirname(process.execPath);
const command = process.argv[2];
const args = process.argv.slice(3);
const usage = 'Usage: node tooling.mjs prepare|install|dev|build|preview|audit|update [arguments]';

function writeIfChanged(path, content) {
  if (!existsSync(path) || readFileSync(path, 'utf8') !== content) {
    writeFileSync(path, content);
  }
}

function validateLock(manifest, lock) {
  const locked = lock.packages?.[''];
  if (!locked || lock.name !== manifest.name || lock.version !== manifest.version ||
      locked.name !== manifest.name || locked.version !== manifest.version ||
      !isDeepStrictEqual(locked.dependencies ?? {}, manifest.dependencies ?? {}) ||
      !isDeepStrictEqual(locked.devDependencies ?? {}, manifest.devDependencies ?? {})) {
    throw new Error('Dependency sources disagree. Run node tooling.mjs update after editing config/package.mjs.');
  }
}

function prepare(validate = true) {
  if (validate) validateLock(packageConfig, packageLock);
  // npm and TypeScript require JSON; keep these generated copies out of Git.
  for (const [name, value] of [
    ['package.json', packageConfig],
    ['package-lock.json', packageLock],
    ['tsconfig.json', typescriptConfig],
  ]) {
    writeIfChanged(join(root, name), JSON.stringify(value, null, 2) + '\n');
  }
}

function npmInvocation() {
  const candidates = [
    join(nodeDirectory, '../lib/node_modules/npm/bin/npm-cli.js'),
    join(nodeDirectory, '../node_modules/npm/bin/npm-cli.js'),
    join(nodeDirectory, 'node_modules/npm/bin/npm-cli.js'),
  ];
  for (const directory of (process.env.PATH ?? '').split(delimiter)) {
    candidates.push(join(directory, 'npm-cli.js'));
    const executable = join(directory, 'npm');
    if (existsSync(executable)) {
      const resolved = realpathSync(executable);
      if (resolved.endsWith('.js')) candidates.push(resolved);
    }
  }
  const cli = candidates.find((path) => existsSync(path));
  return cli ? [process.execPath, [cli]] : ['npm', []];
}

function runNpm(npmArgs) {
  const [executable, prefix] = npmInvocation();
  const result = spawnSync(executable, [...prefix, ...npmArgs], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, PATH: nodeDirectory + delimiter + (process.env.PATH ?? '') },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    if (result.signal) console.error(`npm stopped by ${result.signal}`);
    process.exit(result.status ?? 1);
  }
}

try {
  if (!['prepare', 'install', 'dev', 'build', 'preview', 'audit', 'update'].includes(command)) {
    throw new Error(usage);
  }
  if (command === 'prepare' && args.length) throw new Error(usage);
  prepare(command !== 'update');
  if (command === 'install') runNpm(['ci', ...args]);
  if (['dev', 'build', 'preview'].includes(command)) runNpm(['run', command, ...(args.length ? ['--', ...args] : [])]);
  if (command === 'audit') runNpm(['audit', '--audit-level=moderate', ...args]);
  if (command === 'update') {
    // Explicit maintenance only: retain npm's exact resolved versions and integrity hashes.
    runNpm(['install', '--package-lock-only', '--ignore-scripts', ...args]);
    const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    const lock = JSON.parse(readFileSync(join(root, 'package-lock.json'), 'utf8'));
    validateLock(manifest, lock);
    for (const [name, value] of [['package.mjs', manifest], ['package-lock.mjs', lock]]) {
      writeIfChanged(join(root, 'config', name),
        '// Native module source; generated tool files are ignored.\nexport default ' +
        JSON.stringify(value, null, 2) + ';\n');
    }
    console.log('Updated config/package.mjs and config/package-lock.mjs. Review and commit their changes.');
  }
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
