import * as path from "node:path";
import { Agent } from "@cli/agent";
import { readFile, listFiles } from "@cli/tools";
import { selectProvider } from "@core/providers/factory";
import { REPO_READER_SYSTEM_PROMPT } from "./prompt";

export type RunInput = {
  path: string;
  maxSteps?: number;
  verbose?: boolean;
};

export type RunOutput = {
  summary: string;
};

export async function run(input: RunInput): Promise<RunOutput> {
  const root = path.resolve(input.path);
  process.env.WORKSPACE_ROOT = root;

  const provider = selectProvider();
  const agent = new Agent({
    provider,
    systemPrompt: REPO_READER_SYSTEM_PROMPT,
    tools: [listFiles, readFile],
    maxSteps: input.maxSteps ?? 20,
    verbose: input.verbose ?? false,
  });

  const userPrompt = `Explore the repository rooted at WORKSPACE_ROOT (${root}) and write the architecture brief described in your instructions.`;
  const summary = await agent.run(userPrompt);
  return { summary };
}
