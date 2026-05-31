# ツール設計監査 — Tool Design Audit

> 受領教材「AIエージェントのツール実装」が説く 3 メタデータ規律
> （① スコープを絞った正確な Name ② 他と混同されない Description ③ 厳密な Schema）
> に照らして、`python/langchain/` 配下の全ツール定義を監査する。
> 新規ツールは追加しない。監査と最小改善のみ。

## TL;DR

- ツールは 4 定義。`StructuredTool` 3 本（web_search / search_documents / write_file）と、
  retriever を包む旧式 `Tool` 1 本（wrap_as_tool）。`@tool` デコレータは未使用で、
  すべてビルダー関数が `name` / `description` / `args_schema` を明示的に組み立てる。
- Name と Description は全体に良好。**唯一の構造的な弱点は `wrap_as_tool`**
  で、`StructuredTool` 版と同じ name `search_documents` を持ちながら typed schema を
  公開しない（規律③違反）。
- 挙動を変えずに直せる範囲で、docstring と Field description を補強した。
  schema 自体の差し替え（旧式 Tool への args_schema 付与）は LLM が見る入力面を
  変えるため見送り、代わりに「typed 版を使え」と docstring で誘導した。

## 洗い出した全ツール

| ツール (name) | 定義場所 | 種別 | 引数スキーマ | 利用箇所 |
|---|---|---|---|---|
| `web_search` | `cli/tools/web_search.py` `build_web_search_tool` | StructuredTool | `WebSearchInput { query: str }` | `agents/search_agent/runner.py`, `agents/web_researcher/runner.py` |
| `search_documents` (typed) | `cli/tools/search_documents.py` `build_search_documents_tool` | StructuredTool | `SearchDocumentsInput { query: str }` | `agents/doc_reader/runner.py` |
| `write_file` | `cli/tools/write_file.py` `build_write_file_tool` | StructuredTool (HITL gated) | `WriteFileInput { relative_path: str, content: str }` | `agents/web_researcher/runner.py` |
| `search_documents` (legacy) | `cli/retriever.py` `wrap_as_tool` | Tool（schema なし） | 単一の freeform 文字列のみ | `cli/agent.py` |

### Description 全文

- **web_search**: "Search the public web via Tavily and return ranked results with titles,
  URLs, and content snippets. Use this to gather facts before writing a report.
  Input is a single natural-language query string."
- **search_documents (typed)**: "Search the indexed PDF for passages relevant to a query.
  Use this whenever the user question requires knowledge from the document.
  Input is a single natural-language query string. Output is the top matching chunks
  with page metadata when available."
- **write_file**: "Write text or HTML to a file inside the agent workspace.
  This is a destructive operation: it requires explicit human approval before the bytes
  hit disk. Input is a workspace-relative path and the content."
- **search_documents (legacy)**: "Search the indexed PDF for passages relevant to a query.
  Input must be a focused natural-language query string. Returns the top matching chunks
  with page metadata when available."

## 3 メタデータ監査

| ツール | ① Name | ② Description | ③ Schema | 総評 |
|---|---|---|---|---|
| web_search | ✅ 動詞 + 対象が明確、docs 検索と混同しない | ✅ 取得元（Tavily / public web）・用途（report 前のファクト収集）・入出力を明示 | ✅ `WebSearchInput.query` に Field description あり | 良好 |
| search_documents (typed) | ✅ web_search と対比的でスコープ明確 | ✅ 「PDF 内の知識が要るとき」と使い分け条件を明記 | ✅ typed args_schema + Field description | 良好 |
| write_file | ✅ 操作対象が一意 | ✅ destructive + HITL 必須を前置、危険性を明示 | 🟡→✅ `relative_path` は良好。`content` の説明が弱かったため補強 | 改善済 |
| search_documents (legacy) | 🔴 typed 版と name 衝突 | 🟡 文言自体は妥当だが旧式である旨が無く誤用を招く | 🔴 typed schema を公開せず freeform 文字列のみ | docstring で誘導（schema は不変） |

### 監査メモ

- **name 衝突 (`search_documents` x2)**: 両者を 1 エージェントに同時登録すると LLM 側で
  区別できない。現状は別エージェントで片方ずつ使うため実害はないが、規律①の
  「混同を招かない」観点で潜在リスク。docstring に「never register both in one agent」を明記。
- **legacy の schema 欠落**: `Tool`（`StructuredTool` ではない）は args_schema を持たず、
  LLM には単一 freeform 文字列としてしか見えない。これは規律③（厳密な Schema）の
  明確な違反。挙動を変えずに直す手段が docstring 誘導しかないため、
  `build_search_documents_tool` への移行を促す文言を追加した。
- **builder の `max_results` / `workspace` 引数**: LLM に露出しないビルド時パラメータで、
  ツール schema には含まれない。規律③の対象外で問題なし。

## 改善した箇所（before / after）

挙動（ロジック）は不変。docstring と Field description の文言のみ変更。

### 1. `cli/retriever.py` `wrap_as_tool` docstring

- before: `"""Wrap a retriever as a LangChain Tool that returns a formatted string."""`
- after:

  ```
  Wrap a retriever as a single-input LangChain Tool returning a formatted string.

  This is the schema-less variant: Tool advertises only one freeform string
  argument, so the LLM cannot rely on a named, typed field. Prefer
  cli.tools.search_documents.build_search_documents_tool, which exposes a typed
  pydantic args_schema for structured tool calls. Both share the name
  "search_documents"; never register both in one agent.
  ```

  狙い: 規律③違反であることと、name 衝突リスクを定義箇所で明示し、typed 版へ誘導する。

### 2. `cli/tools/write_file.py` `WriteFileInput.content` の Field description

- before: `"Full text or HTML content to write."`
- after:

  ```
  Complete file contents to write, as UTF-8 text. Pass the entire final document;
  the write overwrites the target, it does not append.
  ```

  狙い: 「全文を渡す」「追記でなく上書き」というエージェントが誤りやすい契約を明文化（規律③のフィールド粒度を強化）。

### 3. `cli/tools/search_documents.py` builder docstring

- before:

  ```
  Unlike the loose Tool wrapper in cli.retriever.wrap_as_tool, this version
  advertises a typed argument schema so the LLM emits structured tool calls.
  ```

- after:

  ```
  Preferred over the schema-less cli.retriever.wrap_as_tool: this version
  advertises a typed args_schema (SearchDocumentsInput) so the LLM emits
  structured tool calls with a named, validated query field.
  ```

  狙い: 2 つの `search_documents` 実装のうち typed 版が正であることを定義箇所で言い切る。

## 改善不要と判断したもの

- **web_search / search_documents (typed) の Description**: 取得元・用途・使い分け条件・
  入出力がそろっており、規律②を満たす。文言いじりは利得が小さいため変更しない。
- **legacy `wrap_as_tool` への args_schema 付与**: schema を付けると LLM が見る入力面が
  変わり「挙動不変」制約に抵触する。本監査では docstring 誘導に留め、移行は別タスク扱い。

## 教材の 3 規律と本リポの対応（面接用まとめ）

- ① Name は「動詞 + 対象」で粒度を 1 ツール 1 責務に寄せる → web/docs/file の 3 系統で衝突回避。
  唯一の負債は同名 2 実装で、これは typed 版への一本化で解消できる。
- ② Description は「何をするか」だけでなく「いつ使うか（条件）」「入出力形式」を書くと
  ツール選択精度が上がる。本リポは web_search / search_documents で実践済み。
- ③ Schema は pydantic で型 + Field description を与えると LLM が構造化呼び出しを出す。
  `Tool` ではなく `StructuredTool` を使うことがこの規律の前提条件。
