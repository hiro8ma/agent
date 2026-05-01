import type {
  AnyTool,
  LanguageModel,
  Message,
  ToolCall,
} from "@core/types";

export type AgentConfig = {
  provider: LanguageModel;
  systemPrompt: string;
  tools: AnyTool[];
  maxSteps?: number;
  verbose?: boolean;
};

export class Agent {
  private readonly provider: LanguageModel;
  private readonly systemPrompt: string;
  private readonly tools: AnyTool[];
  private readonly maxSteps: number;
  private readonly verbose: boolean;

  constructor(config: AgentConfig) {
    this.provider = config.provider;
    this.systemPrompt = config.systemPrompt;
    this.tools = config.tools;
    this.maxSteps = config.maxSteps ?? 10;
    this.verbose = config.verbose ?? false;
  }

  async run(userInput: string): Promise<string> {
    const messages: Message[] = [
      { role: "system", content: this.systemPrompt },
      { role: "user", content: userInput },
    ];

    let lastText = "";
    let step = 0;

    while (step < this.maxSteps) {
      step++;
      if (this.verbose) {
        console.error(`[agent] step ${step}/${this.maxSteps}`);
      }

      const result = await this.provider.doGenerate({
        messages,
        tools: this.tools,
        maxTokens: 4096,
      });

      if (result.text) {
        lastText = result.text;
      }

      const toolCalls = result.toolCalls ?? [];

      if (toolCalls.length === 0) {
        if (result.text) {
          messages.push({ role: "assistant", content: result.text });
        }
        break;
      }

      messages.push({
        role: "assistant",
        content: result.text ?? "",
        toolCalls,
      });

      for (const call of toolCalls) {
        const toolResult = await this.executeTool(call);
        messages.push({
          role: "tool",
          toolCallId: call.id,
          name: call.name,
          content: toolResult,
        });
      }

      if (result.finishReason === "stop") {
        break;
      }
    }

    if (step >= this.maxSteps && this.verbose) {
      console.error(`[agent] reached maxSteps (${this.maxSteps})`);
    }

    return lastText;
  }

  private async executeTool(call: ToolCall): Promise<string> {
    const tool = this.tools.find((t) => t.name === call.name);
    if (!tool) {
      return `error: unknown tool ${call.name}`;
    }
    if (this.verbose) {
      console.error(`[tool] ${call.name}(${JSON.stringify(call.args)})`);
    }
    try {
      const parsed = tool.parameters.safeParse(call.args);
      if (!parsed.success) {
        return `error: invalid arguments: ${parsed.error.message}`;
      }
      const out = await tool.execute(parsed.data);
      return out;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return `error: ${msg}`;
    }
  }
}
