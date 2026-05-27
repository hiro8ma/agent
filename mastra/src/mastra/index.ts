import { Mastra } from "@mastra/core";
import { PinoLogger } from "@mastra/loggers";
import { assistant } from "./agents/assistant";

export const mastra = new Mastra({
  agents: { assistant },
  logger: new PinoLogger({ name: "agent-mastra", level: "info" }),
});
