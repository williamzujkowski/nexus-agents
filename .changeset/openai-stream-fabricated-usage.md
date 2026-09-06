---
'nexus-agents': patch
---

An OpenAI stream no longer reports a usage block in which nothing was measured.
OpenAI populates `chunk.usage` on a streaming response only when the request
sets `stream_options: { include_usage: true }`, which nothing in this tree does,
so both `?? 0` fallbacks always took the zero branch and every stream emitted
`{inputTokens: 0, outputTokens: 0, totalTokens: 0}`. `inputTokensMeasured:
false` covered only the first of the three, and there is no
`outputTokensMeasured` — so a consumer honouring the flag discounted
`inputTokens` and read `outputTokens: 0` as a measured zero. `usage` is now
omitted when the API reported none, matching the SDK adapter's stream path and
the policy documented on the field itself.
