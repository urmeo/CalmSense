#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync, realpathSync, writeFileSync } from 'node:fs';
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
const usage = 'Usage: node tooling.mjs prepare|install|test|dev|build|preview|audit|update [arguments]';

function writeIfChanged(path, content) {
  if (!existsSync(path) || readFileSync(path, 'utf8') !== content) {
    writeFileSync(path, content);
  }
}

function lockSource(lock) {
  const { packages, ...metadata } = lock;
  const entries = Object.entries(packages).map(([name, value]) =>
    JSON.stringify(name) + ':' + JSON.stringify(value));
  return 'export default {\n' + JSON.stringify(metadata).slice(1, -1) +
    ',\n"packages":{\n' + entries.join(',\n') + '\n}\n};\n';
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
  // Required tool JSON.
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

function runNode(nodeArgs) {
  const result = spawnSync(process.execPath, nodeArgs, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

try {
  if (!['prepare', 'install', 'test', 'dev', 'build', 'preview', 'audit', 'update'].includes(command)) {
    throw new Error(usage);
  }
  if (command === 'prepare' && args.length) throw new Error(usage);
  prepare(command !== 'update');
  if (command === 'install') runNpm(['ci', ...args]);
  if (command === 'test') {
    const tests = readdirSync(join(root, 'tests')).filter((name) => name.endsWith('.test.mjs'));
    runNode(['--experimental-strip-types', '--test', ...args, ...tests.map((name) => join('tests', name))]);
  }
  if (command === 'build') runNode(['node_modules/typescript/bin/tsc']);
  if (['dev', 'build', 'preview'].includes(command)) runNode(['build.mjs', command, ...args]);
  if (command === 'audit') runNpm(['audit', '--audit-level=moderate', ...args]);
  if (command === 'update') {
    runNpm(['install', '--package-lock-only', '--ignore-scripts', ...args]);
    const manifest = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
    const lock = JSON.parse(readFileSync(join(root, 'package-lock.json'), 'utf8'));
    validateLock(manifest, lock);
    for (const [name, value] of [['package.mjs', manifest], ['package-lock.mjs', lock]]) {
      writeIfChanged(join(root, 'config', name),
        name === 'package-lock.mjs' ? lockSource(value) :
          'export default ' + JSON.stringify(value, null, 2) + ';\n');
    }
    console.log('Updated config/package.mjs and config/package-lock.mjs. Review and commit their changes.');
  }
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
