import { describe, expect, test } from "bun:test";
import {
  buildUsageRecord,
  estimateCost,
  lookupPricing,
  UsageLogger,
  type UsageRecord,
} from "@core/observability";

describe("lookupPricing", () => {
  test("完全一致で単価を引く", () => {
    expect(lookupPricing("gpt-5-mini")?.input).toBe(0.25);
  });

  test("プレフィックス最長一致で版差を吸収する", () => {
    expect(lookupPricing("claude-sonnet-4-6")?.output).toBe(15.0);
    expect(lookupPricing("gpt-5-mini-2026-01")?.input).toBe(0.25);
  });

  test("未知モデルは undefined", () => {
    expect(lookupPricing("mystery-model")).toBeUndefined();
  });
});

describe("estimateCost", () => {
  test("input/output 単価で概算する", () => {
    // gpt-5-mini: input 0.25 / output 2.0 per 1M
    const cost = estimateCost("gpt-5-mini", {
      inputTokens: 1_000_000,
      outputTokens: 1_000_000,
      cachedInputTokens: 0,
    });
    expect(cost).toBeCloseTo(0.25 + 2.0, 6);
  });

  test("cached 分は input から差し引いて割引単価で計算する", () => {
    // 1M input のうち 1M すべて cached → cachedInput 0.025
    const cost = estimateCost("gpt-5-mini", {
      inputTokens: 1_000_000,
      outputTokens: 0,
      cachedInputTokens: 1_000_000,
    });
    expect(cost).toBeCloseTo(0.025, 6);
  });

  test("未知モデルは null", () => {
    expect(
      estimateCost("mystery", {
        inputTokens: 100,
        outputTokens: 100,
        cachedInputTokens: 0,
      }),
    ).toBeNull();
  });
});

describe("buildUsageRecord", () => {
  test("Usage を OTel 寄りの属性名に正規化しコストを付与する", () => {
    const record = buildUsageRecord({
      provider: "openai",
      model: "gpt-5-mini",
      operation: "chat",
      traceId: "t-1",
      usage: {
        promptTokens: 1_000_000,
        completionTokens: 1_000_000,
        cachedInputTokens: 0,
        reasoningTokens: 200,
        totalTokens: 2_000_000,
      },
    });
    expect(record["gen_ai.system"]).toBe("openai");
    expect(record["gen_ai.request.model"]).toBe("gpt-5-mini");
    expect(record["gen_ai.usage.input_tokens"]).toBe(1_000_000);
    expect(record["gen_ai.usage.output_tokens"]).toBe(1_000_000);
    expect(record["gen_ai.usage.reasoning_tokens"]).toBe(200);
    expect(record["gen_ai.usage.cost_usd"]).toBeCloseTo(0.25 + 2.0, 6);
    expect(record["trace.id"]).toBe("t-1");
  });

  test("単価未知なら cost を省略する", () => {
    const record = buildUsageRecord({
      provider: "x",
      model: "mystery",
      usage: { promptTokens: 10, completionTokens: 10 },
    });
    expect(record["gen_ai.usage.cost_usd"]).toBeUndefined();
  });

  test("usage 欠損でも 0 埋めで record を作る", () => {
    const record = buildUsageRecord({
      provider: "x",
      model: "mystery",
      usage: undefined,
    });
    expect(record["gen_ai.usage.input_tokens"]).toBe(0);
    expect(record["gen_ai.usage.total_tokens"]).toBe(0);
  });
});

describe("UsageLogger", () => {
  test("sink に record を流す", () => {
    const records: UsageRecord[] = [];
    const logger = new UsageLogger((r) => records.push(r));
    logger.log({
      provider: "openai",
      model: "gpt-5-mini",
      usage: { promptTokens: 100, completionTokens: 50 },
    });
    expect(records).toHaveLength(1);
    expect(records[0]!["gen_ai.usage.input_tokens"]).toBe(100);
  });
});
