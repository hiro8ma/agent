import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { MetricsRecorder, type MetricsRecord } from "@core/metrics";

let tmpDir: string;

beforeEach(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "agent-metrics-"));
});

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true });
});

describe("MetricsRecorder", () => {
  test("appends a JSONL line with all 3 layers", async () => {
    const file = path.join(tmpDir, "sub", "runs.jsonl");
    const recorder = new MetricsRecorder(file);

    await recorder.recordRun("trace-1", {
      product: { satisfaction: 4 },
      system: { latencyMs: 1234, tokenUsage: { totalTokens: 500 } },
      aiProxy: { responseQualityScore: 5, hallucinationFlag: false },
    });
    await recorder.recordRun("trace-2", {
      system: { latencyMs: 800 },
    });

    const raw = await fs.readFile(file, "utf8");
    const lines = raw.trimEnd().split("\n");
    expect(lines).toHaveLength(2);

    const first = JSON.parse(lines[0]!) as MetricsRecord;
    expect(first.traceId).toBe("trace-1");
    expect(first.product?.satisfaction).toBe(4);
    expect(first.system?.tokenUsage?.totalTokens).toBe(500);
    expect(first.aiProxy?.responseQualityScore).toBe(5);
    expect(typeof first.timestamp).toBe("string");
    expect(() => new Date(first.timestamp).toISOString()).not.toThrow();

    const second = JSON.parse(lines[1]!) as MetricsRecord;
    expect(second.traceId).toBe("trace-2");
    expect(second.product).toBeUndefined();
    expect(second.aiProxy).toBeUndefined();
  });
});
