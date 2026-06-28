import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { editFile, grep, listFiles, readFile, writeFile } from "@cli/tools";

let workspace: string;
let prevRoot: string | undefined;

beforeEach(async () => {
  workspace = await fs.mkdtemp(path.join(os.tmpdir(), "agent-tools-"));
  prevRoot = process.env.WORKSPACE_ROOT;
  process.env.WORKSPACE_ROOT = workspace;
  await fs.writeFile(path.join(workspace, "a.txt"), "hello world\nsecond line\n");
  await fs.mkdir(path.join(workspace, "sub"));
  await fs.writeFile(path.join(workspace, "sub", "b.ts"), "const HELLO = 1;\n");
});

afterEach(async () => {
  if (prevRoot === undefined) delete process.env.WORKSPACE_ROOT;
  else process.env.WORKSPACE_ROOT = prevRoot;
  await fs.rm(workspace, { recursive: true, force: true });
});

describe("readFile", () => {
  test("reads a file inside the workspace", async () => {
    expect(await readFile.execute({ path: "a.txt" })).toContain("hello world");
  });

  test("rejects paths outside the workspace", async () => {
    await expect(readFile.execute({ path: "../escape.txt" })).rejects.toThrow(
      /outside workspace/,
    );
  });
});

describe("listFiles", () => {
  test("lists entries with directories suffixed by '/'", async () => {
    const out = await listFiles.execute({ path: ".", recursive: true });
    expect(out).toContain("a.txt");
    expect(out).toContain("sub/");
    expect(out).toContain(path.join("sub", "b.ts"));
  });
});

describe("grep", () => {
  test("matches a pattern and reports path:line:text", async () => {
    const out = await grep.execute({ pattern: "second" });
    expect(out).toBe("a.txt:2:second line");
  });

  test("case-insensitive search finds matches across files", async () => {
    const out = await grep.execute({ pattern: "hello", ignoreCase: true });
    expect(out).toContain("a.txt:1:hello world");
    expect(out).toContain(`${path.join("sub", "b.ts")}:1:const HELLO = 1;`);
  });

  test("returns (no matches) when nothing matches", async () => {
    expect(await grep.execute({ pattern: "zzz-nope" })).toBe("(no matches)");
  });

  test("rejects an invalid regular expression", async () => {
    await expect(grep.execute({ pattern: "(" })).rejects.toThrow(
      /invalid regular expression/,
    );
  });
});

describe("writeFile", () => {
  test("is marked as requiring approval", () => {
    expect(writeFile.needsApproval).toBe(true);
  });

  test("creates a file and parent directories", async () => {
    const out = await writeFile.execute({
      path: "nested/dir/new.txt",
      content: "data",
    });
    expect(out).toContain("wrote 4 bytes");
    const written = await fs.readFile(
      path.join(workspace, "nested", "dir", "new.txt"),
      "utf-8",
    );
    expect(written).toBe("data");
  });

  test("overwrites an existing file", async () => {
    await writeFile.execute({ path: "a.txt", content: "replaced" });
    expect(await readFile.execute({ path: "a.txt" })).toBe("replaced");
  });

  test("rejects writes outside the workspace", async () => {
    await expect(
      writeFile.execute({ path: "../escape.txt", content: "x" }),
    ).rejects.toThrow(/outside workspace/);
  });
});

describe("editFile", () => {
  test("is marked as requiring approval", () => {
    expect(editFile.needsApproval).toBe(true);
  });

  test("replaces a unique snippet", async () => {
    const out = await editFile.execute({
      path: "a.txt",
      oldText: "second line",
      newText: "edited line",
    });
    expect(out).toContain("1 replacement");
    expect(await readFile.execute({ path: "a.txt" })).toBe(
      "hello world\nedited line\n",
    );
  });

  test("rejects when oldText is not found", async () => {
    await expect(
      editFile.execute({ path: "a.txt", oldText: "missing", newText: "x" }),
    ).rejects.toThrow(/no match/);
  });

  test("rejects when oldText matches multiple times", async () => {
    await fs.writeFile(path.join(workspace, "dup.txt"), "x\nx\n");
    await expect(
      editFile.execute({ path: "dup.txt", oldText: "x", newText: "y" }),
    ).rejects.toThrow(/matched 2 times/);
  });

  test("treats replacement special patterns literally", async () => {
    await editFile.execute({
      path: "a.txt",
      oldText: "hello world",
      newText: "$& and $1 stay literal",
    });
    expect(await readFile.execute({ path: "a.txt" })).toBe(
      "$& and $1 stay literal\nsecond line\n",
    );
  });

  test("rejects an empty oldText", async () => {
    await expect(
      editFile.execute({ path: "a.txt", oldText: "", newText: "x" }),
    ).rejects.toThrow(/must not be empty/);
  });

  test("reports file not found", async () => {
    await expect(
      editFile.execute({ path: "nope.txt", oldText: "a", newText: "b" }),
    ).rejects.toThrow(/file not found/);
  });

  test("rejects edits outside the workspace", async () => {
    await expect(
      editFile.execute({ path: "../escape.txt", oldText: "a", newText: "b" }),
    ).rejects.toThrow(/outside workspace/);
  });
});
