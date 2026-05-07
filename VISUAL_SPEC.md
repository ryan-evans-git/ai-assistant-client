# VisualSpec — wire contract for assistant-rendered visuals

This document is the **single source of truth** for the data shape
exchanged between `ai-assistant-client` (Python, server) and
`ai-assistant-ui` (TypeScript, client).  Both repos commit an
identical copy.  Changes here require a coordinated bump of
`schema_version` on both sides.

The LLM never produces a `VisualSpec` block directly.  Instead it
calls the `render_visual(kind, spec)` meta-tool — the agent loop
validates the spec, emits an `AgentEvent("visual", {schema_version,
spec})`, and persists a `VisualBlock` in the assistant message in
history.

## Top-level envelope

```json
{
  "schema_version": 1,
  "spec": <VisualSpec>
}
```

## VisualSpec — discriminated union (`kind` field)

### `kind: "chart"`

```jsonc
{
  "kind": "chart",
  "chart_type": "bar" | "line" | "area" | "pie" | "donut" | "scatter",
  "title": "Optional title",        // optional
  "data": [                         // array of row objects
    {"month": "Jan", "revenue": 12000, "cost": 8200},
    {"month": "Feb", "revenue": 14500, "cost": 8400}
  ],
  "x_key": "month",                 // required for non-pie/donut
  "y_keys": ["revenue", "cost"],    // required for non-pie/donut; multiple = grouped/stacked
  "x_label": "Month",               // optional
  "y_label": "USD",                 // optional
  "stacked": false                  // optional, default false; bar/area only
}
```

For `chart_type` = `"pie"` | `"donut"`:

```jsonc
{
  "kind": "chart",
  "chart_type": "pie",
  "title": "Revenue by region",
  "data": [
    {"label": "North", "value": 12000},
    {"label": "South",  "value": 8500}
  ]
  // x_key/y_keys ignored; rows must have {label, value}
}
```

### `kind: "table"`

```jsonc
{
  "kind": "table",
  "title": "Outstanding invoices",   // optional
  "columns": [
    {"key": "id",     "label": "ID",       "type": "string"},
    {"key": "amount", "label": "Amount",   "type": "currency", "align": "right"},
    {"key": "due",    "label": "Due date", "type": "date"}
  ],
  "rows": [
    {"id": "4711", "amount": 1250.00, "due": "2026-03-15"},
    {"id": "4733", "amount":  890.00, "due": "2026-03-22"}
  ]
}
```

`columns[].type` is a hint to the renderer for formatting:
`"string" | "number" | "currency" | "date" | "boolean"`.
`columns[].align` is `"left" | "center" | "right"` (optional).

### `kind: "kpi"`

A single big stat with optional trend.

```jsonc
{
  "kind": "kpi",
  "label": "Open invoices > 30 days past due",
  "value": "$4,340.00",
  "unit": null,                              // optional, displayed after value
  "trend": {                                 // optional
    "direction": "up" | "down" | "flat",
    "delta": "+12%",
    "period": "vs. last month"               // optional
  },
  "status": "warn"                           // optional: "good"|"warn"|"bad"|"neutral"
}
```

### `kind: "image"`

```jsonc
{
  "kind": "image",
  "src": "https://example.com/chart.png",    // see Image src safety below
  "alt": "Bar chart of revenue by quarter",
  "width": 600,                              // optional
  "height": 400,                             // optional
  "caption": "Source: 2026 Q1 board deck"    // optional
}
```

#### Image `src` safety

Both sides enforce the same allowlist:

- `https://...` — accepted.  HTTP and other schemes (`file://`,
  `javascript:`, `ftp://`) rejected.
- `data:image/(png|jpeg|jpg|gif|webp);base64,<base64>` — accepted.
  Used for previewing local uploads from the host application.
- `data:image/svg+xml;...` — **rejected**.  SVG can carry script.
- `data:` URIs are size-capped at **5 MB** of decoded content
  (configurable via `AgentRunConfig.max_image_data_uri_kb`).

Reject in the agent's `render_visual` handler with a clear error in
the tool_result, so the model can correct or fall back to text.  The
UI also re-validates defensively before rendering — never trust the
network.

## Validation

- Server: Pydantic models in `ai_assistant_client/visuals/types.py`.
- UI: Zod schemas in `src/visuals/schemas.ts`.
- Both validate at the wire boundary; a malformed spec from the LLM
  is surfaced as a `tool_result` error to the model so it can revise.

## Versioning

`schema_version` is an integer.  v1 is described above.  Breaking
changes bump to v2 and require coordinated release of both repos.
The UI ignores any envelope whose `schema_version` it doesn't
understand and renders a graceful fallback.

## Out of scope for v1

Maps, sparklines, code blocks (already rendered as markdown),
interactive drill-down, animation timelines, real-time streaming
data updates.
