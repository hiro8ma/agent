import { describe, expect, test } from "bun:test";
import {
  isModelTier,
  isProviderName,
  resolveModel,
} from "@core/providers/models";

describe("resolveModel", () => {
  test("生のモデル ID が指定されたら tier より優先する", () => {
    expect(
      resolveModel("anthropic", { modelId: "claude-3-legacy", tier: "smart" }),
    ).toBe("claude-3-legacy");
  });

  test("modelId 未指定なら tier でテーブルから解決する", () => {
    expect(resolveModel("anthropic", { tier: "fast" })).toBe(
      "claude-haiku-4-5-20251001",
    );
    expect(resolveModel("anthropic", { tier: "default" })).toBe(
      "claude-sonnet-4-6",
    );
    expect(resolveModel("anthropic", { tier: "smart" })).toBe(
      "claude-opus-4-7",
    );
  });

  test("tier 未指定なら default にフォールバックする", () => {
    expect(resolveModel("openai")).toBe("gpt-5");
    expect(resolveModel("google")).toBe("gemini-2.5-flash");
    expect(resolveModel("anthropic")).toBe("claude-sonnet-4-6");
  });

  test("provider ごとに tier テーブルが整合する", () => {
    expect(resolveModel("openai", { tier: "fast" })).toBe("gpt-5-mini");
    expect(resolveModel("openai", { tier: "smart" })).toBe("gpt-5");
    expect(resolveModel("google", { tier: "smart" })).toBe("gemini-2.5-pro");
  });

  test("空白のみの modelId は無視して tier で解決する", () => {
    expect(resolveModel("google", { modelId: "  ", tier: "smart" })).toBe(
      "gemini-2.5-pro",
    );
  });

  test("modelId が undefined でも tier で解決する", () => {
    expect(resolveModel("openai", { modelId: undefined, tier: "fast" })).toBe(
      "gpt-5-mini",
    );
  });
});

describe("isModelTier", () => {
  test("既知の tier を受理する", () => {
    expect(isModelTier("fast")).toBe(true);
    expect(isModelTier("default")).toBe(true);
    expect(isModelTier("smart")).toBe(true);
  });

  test("不正な tier を拒否する", () => {
    expect(isModelTier("turbo")).toBe(false);
    expect(isModelTier("")).toBe(false);
    expect(isModelTier("Fast")).toBe(false);
  });
});

describe("isProviderName", () => {
  test("既知の provider を受理する", () => {
    expect(isProviderName("anthropic")).toBe(true);
    expect(isProviderName("openai")).toBe(true);
    expect(isProviderName("google")).toBe(true);
  });

  test("不明な provider を拒否する", () => {
    expect(isProviderName("mistral")).toBe(false);
    expect(isProviderName("")).toBe(false);
  });
});
