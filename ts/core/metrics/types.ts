// 3 階層メトリクス型定義
//
// 出典: 「AIプロダクトの成功測定」3 階層モデル
//   - Product   ユーザー価値（engagement / satisfaction / adoption / conversion / retention）
//   - System    技術品質（latency / uptime / error_rate / token_usage）
//   - AI Proxy  AI 固有品質（response_quality / citation_accuracy / hallucination / tool_success）
//
// 設計方針
//   - 3 階層は独立に記録できるよう各フィールドを optional にする
//   - LLM-as-judge のスコアは 1-5 を共通スケールにする（後で集計しやすい）
//   - core/types.ts の Usage と整合させ、system 層の token_usage はそのまま再利用する

import type { Usage } from "../types";

// プロダクトメトリクス（ユーザー価値層）
// 1 回の run で全てが埋まることは稀で、後段の集計で時間軸の指標へ変換する
export type ProductMetric = {
  engagement?: number | undefined;     // 0-1 セッション継続度
  satisfaction?: number | undefined;   // 1-5 ユーザー評価（CSAT 相当）
  adoption?: boolean | undefined;      // 機能を実際に使い切ったか
  conversion?: boolean | undefined;    // 目的アクション（購入 / 問い合わせ）達成
  retention?: boolean | undefined;     // リピート利用
};

// システムメトリクス（技術層）
export type SystemMetric = {
  latencyMs?: number | undefined;      // 1 リクエストの所要時間
  uptime?: number | undefined;         // 0-1 期間内の可用率
  errorRate?: number | undefined;      // 0-1 期間内のエラー率
  tokenUsage?: Usage | undefined;      // core/types.ts の Usage をそのまま使う
};

// AI 代替メトリクス（AI Proxy 層）
// AI 出力の品質を間接的に観測するための代理指標
export type AIProxyMetric = {
  responseQualityScore?: number | undefined;  // 1-5 LLM-as-judge の総合スコア
  citationAccuracy?: number | undefined;      // 0-1 引用の事実整合率
  hallucinationFlag?: boolean | undefined;    // 幻覚と判定されたか
  toolSuccessRate?: number | undefined;       // 0-1 セッション内の tool 成功率
};

// 1 件の run に紐づくメトリクスレコード
// traceId は分散トレーシング側と突合できるよう必須
export type MetricsRecord = {
  traceId: string;
  timestamp: string;                        // ISO 8601
  product?: ProductMetric | undefined;
  system?: SystemMetric | undefined;
  aiProxy?: AIProxyMetric | undefined;
};

// LLM-as-judge のスコア詳細
// judge プロンプトと 1:1 で対応させる
export type JudgeScore = {
  faithfulness: number;   // 1-5 与えられた context に忠実か
  relevance: number;      // 1-5 質問にどれだけ答えているか
  conciseness: number;    // 1-5 冗長でないか
  overall: number;        // 1-5 総合（上記の平均ではなく judge が独立に出す）
  rationale: string;      // 1-2 文の判定理由
};
