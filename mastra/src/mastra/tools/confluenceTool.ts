import { createTool } from "@mastra/core/tools";
import { z } from "zod";

// Confluence Cloud REST API への接続。認証情報は環境変数から読む。
//   CONFLUENCE_BASE_URL    例: https://your-domain.atlassian.net/wiki
//   CONFLUENCE_USER_EMAIL  Atlassian アカウントのメール
//   CONFLUENCE_API_TOKEN   API トークン

function getAuthHeaders(): Record<string, string> {
  const email = process.env.CONFLUENCE_USER_EMAIL ?? "";
  const token = process.env.CONFLUENCE_API_TOKEN ?? "";
  const basic = Buffer.from(`${email}:${token}`).toString("base64");
  return {
    Authorization: `Basic ${basic}`,
    Accept: "application/json",
  };
}

async function callConfluenceAPI(path: string): Promise<any> {
  const baseUrl = process.env.CONFLUENCE_BASE_URL;
  if (!baseUrl) {
    throw new Error("CONFLUENCE_BASE_URL is not set");
  }
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) {
    // 認証情報・権限切れを呼び出し側で構造化エラーにできるよう本文を含めて投げる。
    const body = await response.text();
    throw new Error(`Confluence API ${response.status} ${response.statusText}: ${body}`);
  }
  return response.json();
}

export const confluenceSearchPagesTool = createTool({
  id: "confluence-search-pages",
  description:
    "CQL（Confluence Query Language）で Confluence ページを検索し、ヒットしたページの一覧を返す。",
  inputSchema: z.object({
    cql: z
      .string()
      .describe('CQL クエリ。例: text ~ "設計" and type = page'),
  }),
  outputSchema: z.object({
    pages: z.array(
      z.object({
        id: z.string(),
        title: z.string(),
      }),
    ),
    total: z.number(),
    error: z.string().optional(),
  }),
  execute: async ({ cql }) => {
    try {
      const data = await callConfluenceAPI(
        `/rest/api/content/search?cql=${encodeURIComponent(cql)}`,
      );
      const results: any[] = data.results ?? [];
      return {
        pages: results.map((r) => ({ id: String(r.id), title: String(r.title) })),
        total: Number(data.size ?? results.length),
      };
    } catch (e) {
      return {
        pages: [],
        total: 0,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  },
});

export const confluenceGetPageTool = createTool({
  id: "confluence-get-page",
  description:
    "指定された ID の Confluence ページの本文（HTML）を取得する。",
  inputSchema: z.object({
    pageId: z.string().describe("取得対象の Confluence ページ ID"),
    expand: z
      .string()
      .default("body.storage")
      .describe("展開するフィールド。既定は本文 HTML（body.storage）"),
  }),
  outputSchema: z.object({
    id: z.string(),
    title: z.string(),
    body: z.string(),
    error: z.string().optional(),
  }),
  execute: async ({ pageId, expand }) => {
    try {
      const expandFields = expand ?? "body.storage";
      const data = await callConfluenceAPI(
        `/rest/api/content/${encodeURIComponent(pageId)}?expand=${encodeURIComponent(expandFields)}`,
      );
      return {
        id: String(data.id),
        title: String(data.title),
        body: String(data.body?.storage?.value ?? ""),
      };
    } catch (e) {
      return {
        id: pageId,
        title: "",
        body: "",
        error: e instanceof Error ? e.message : String(e),
      };
    }
  },
});
