#!/usr/bin/env bun
import { run } from "@agents/repo-reader/runner";

type Args = {
  path?: string;
  verbose?: boolean;
  help?: boolean;
};

function parseArgs(argv: string[]): Args {
  const args: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--path" || a === "-p") {
      const next = argv[++i];
      if (next !== undefined) args.path = next;
    } else if (a === "--verbose" || a === "-v") {
      args.verbose = true;
    } else if (a === "--help" || a === "-h") {
      args.help = true;
    }
  }
  return args;
}

function printUsage(): void {
  console.log(`repo-reader — explore a repo and emit an architecture brief.

Usage:
  bun run bin/repo-reader.ts --path <dir> [--verbose]

Options:
  --path, -p     Path to the repository to explore (required)
  --verbose, -v  Print step / tool trace to stderr
  --help, -h     Show this message

Environment:
  ANTHROPIC_API_KEY  Required.
  LLM_PROVIDER       Default: anthropic
  LLM_MODEL          Default: claude-sonnet-4-5-20250929
`);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printUsage();
    return;
  }
  if (!args.path) {
    console.error("error: --path is required");
    printUsage();
    process.exit(1);
  }
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("error: ANTHROPIC_API_KEY is not set");
    process.exit(1);
  }

  try {
    const out = await run({
      path: args.path,
      ...(args.verbose ? { verbose: true } : {}),
    });
    console.log(out.summary);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`error: ${msg}`);
    process.exit(1);
  }
}

await main();
