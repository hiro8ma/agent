// 3 プロバイダー（OpenAI / Anthropic / Google）の API を SDK を使わず raw fetch で呼ぶデモ。
// 通信の詳細を理解して core/providers/*.ts の SDK 抽象化が何を担当しているかを体感する用。
//
// 使い方
//   bun run playground/simple-call.ts anthropic
//   bun run playground/simple-call.ts openai
//   bun run playground/simple-call.ts google

const PROMPT = "TypeScript について簡潔に説明してください。";

async function callOpenAI(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error("OPENAI_API_KEY is not set");

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "gpt-5-mini",
      messages: [{ role: "user", content: PROMPT }],
    }),
  });

  if (!response.ok) {
    throw new Error(`OpenAI API error: ${response.status} ${await response.text()}`);
  }
  const data = (await response.json()) as {
    choices: Array<{ message: { content: string }; finish_reason: string }>;
    usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  };
  console.log("[content]", data.choices[0]?.message.content ?? "");
  console.log("[finish_reason]", data.choices[0]?.finish_reason);
  console.log("[usage]", data.usage);
}

async function callAnthropic(): Promise<void> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 1024,
      messages: [{ role: "user", content: PROMPT }],
    }),
  });

  if (!response.ok) {
    throw new Error(`Anthropic API error: ${response.status} ${await response.text()}`);
  }
  const data = (await response.json()) as {
    content: Array<{ type: string; text?: string }>;
    stop_reason: string;
    usage: { input_tokens: number; output_tokens: number };
  };
  const text = data.content
    .filter((b) => b.type === "text")
    .map((b) => b.text ?? "")
    .join("");
  console.log("[content]", text);
  console.log("[stop_reason]", data.stop_reason);
  console.log("[usage]", data.usage);
}

async function callGoogle(): Promise<void> {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) throw new Error("GOOGLE_API_KEY is not set");

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: PROMPT }] }],
    }),
  });

  if (!response.ok) {
    throw new Error(`Google API error: ${response.status} ${await response.text()}`);
  }
  const data = (await response.json()) as {
    candidates: Array<{
      content: { parts: Array<{ text?: string }> };
      finishReason: string;
    }>;
    usageMetadata?: {
      promptTokenCount: number;
      candidatesTokenCount: number;
      totalTokenCount: number;
    };
  };
  const text = (data.candidates[0]?.content.parts ?? [])
    .map((p) => p.text ?? "")
    .join("");
  console.log("[content]", text);
  console.log("[finish_reason]", data.candidates[0]?.finishReason);
  console.log("[usage]", data.usageMetadata);
}

const provider = process.argv[2] ?? "anthropic";
const fns: Record<string, () => Promise<void>> = {
  openai: callOpenAI,
  anthropic: callAnthropic,
  google: callGoogle,
};

const fn = fns[provider];
if (!fn) {
  console.error(
    `Unknown provider: ${provider}. Use one of: openai / anthropic / google`,
  );
  process.exit(1);
}

await fn();
