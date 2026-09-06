/**
 * Tests for OpenAI Message Mappers
 * @module adapters/openai-mappers.test
 */

import { describe, it, expect } from 'vitest';
import type { Message, ToolDefinition } from '../core/index.js';
import type { ChatCompletion, ChatCompletionChunk } from 'openai/resources/chat/completions';
import {
  mapStopReason,
  mapChoiceToContentBlocks,
  mapMessage,
  mapTool,
  mapResponseUsage,
  mapStreamChunk,
} from './openai-mappers.js';

// ============================================================================
// mapStopReason
// ============================================================================

describe('mapStopReason', () => {
  it('maps stop to end_turn', () => {
    expect(mapStopReason('stop')).toBe('end_turn');
  });

  it('maps length to max_tokens', () => {
    expect(mapStopReason('length')).toBe('max_tokens');
  });

  it('maps tool_calls to tool_use', () => {
    expect(mapStopReason('tool_calls')).toBe('tool_use');
  });

  it('maps function_call to tool_use', () => {
    expect(mapStopReason('function_call')).toBe('tool_use');
  });

  it('maps content_filter to end_turn', () => {
    expect(mapStopReason('content_filter')).toBe('end_turn');
  });

  it('maps null to end_turn', () => {
    expect(mapStopReason(null)).toBe('end_turn');
  });

  it('maps undefined to end_turn', () => {
    expect(mapStopReason(undefined)).toBe('end_turn');
  });

  it('maps unknown string to end_turn', () => {
    expect(mapStopReason('unknown_reason')).toBe('end_turn');
  });
});

// ============================================================================
// mapChoiceToContentBlocks
// ============================================================================

describe('mapChoiceToContentBlocks', () => {
  it('maps text content', () => {
    const choice = {
      message: { content: 'Hello world', role: 'assistant' as const },
      index: 0,
      finish_reason: 'stop' as const,
      logprobs: null,
    } as ChatCompletion.Choice;
    const blocks = mapChoiceToContentBlocks(choice);
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({ type: 'text', text: 'Hello world' });
  });

  it('returns empty text block when no content', () => {
    const choice = {
      message: { content: null, role: 'assistant' as const },
      index: 0,
      finish_reason: 'stop' as const,
      logprobs: null,
    } as ChatCompletion.Choice;
    const blocks = mapChoiceToContentBlocks(choice);
    expect(blocks).toHaveLength(1);
    expect(blocks[0]).toEqual({ type: 'text', text: '' });
  });

  it('maps tool calls', () => {
    const choice = {
      message: {
        content: null,
        role: 'assistant' as const,
        tool_calls: [
          {
            id: 'call-1',
            type: 'function' as const,
            function: { name: 'get_weather', arguments: '{"city":"NYC"}' },
          },
        ],
      },
      index: 0,
      finish_reason: 'tool_calls' as const,
      logprobs: null,
    } as ChatCompletion.Choice;
    const blocks = mapChoiceToContentBlocks(choice);
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.type).toBe('tool_use');
  });

  it('handles invalid JSON in tool call arguments', () => {
    const choice = {
      message: {
        content: null,
        role: 'assistant' as const,
        tool_calls: [
          {
            id: 'call-1',
            type: 'function' as const,
            function: { name: 'test', arguments: '{invalid json' },
          },
        ],
      },
      index: 0,
      finish_reason: 'tool_calls' as const,
      logprobs: null,
    } as ChatCompletion.Choice;
    const blocks = mapChoiceToContentBlocks(choice);
    expect(blocks).toHaveLength(1);
    expect(blocks[0]?.type).toBe('tool_use');
    // Should fallback to raw string
    const toolBlock = blocks[0] as { type: 'tool_use'; input: { _raw: string } };
    expect(toolBlock.input._raw).toBe('{invalid json');
  });

  it('maps both text and tool calls', () => {
    const choice = {
      message: {
        content: 'Let me check',
        role: 'assistant' as const,
        tool_calls: [
          {
            id: 'call-1',
            type: 'function' as const,
            function: { name: 'search', arguments: '{}' },
          },
        ],
      },
      index: 0,
      finish_reason: 'tool_calls' as const,
      logprobs: null,
    } as ChatCompletion.Choice;
    const blocks = mapChoiceToContentBlocks(choice);
    expect(blocks).toHaveLength(2);
    expect(blocks[0]?.type).toBe('text');
    expect(blocks[1]?.type).toBe('tool_use');
  });
});

// ============================================================================
// mapMessage
// ============================================================================

describe('mapMessage', () => {
  it('maps system message with string content', () => {
    const message: Message = { role: 'system', content: 'You are helpful' };
    const result = mapMessage(message);
    expect(result.role).toBe('system');
    expect(result.content).toBe('You are helpful');
  });

  it('maps system message with content blocks', () => {
    const message: Message = {
      role: 'system',
      content: [
        { type: 'text', text: 'Part 1' },
        { type: 'text', text: 'Part 2' },
      ],
    };
    const result = mapMessage(message);
    expect(result.role).toBe('system');
    expect(result.content).toBe('Part 1\nPart 2');
  });

  it('maps user message with string content', () => {
    const message: Message = { role: 'user', content: 'Hello' };
    const result = mapMessage(message);
    expect(result.role).toBe('user');
    expect(result.content).toBe('Hello');
  });

  it('maps user message with tool result', () => {
    const message: Message = {
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: 'call-1',
          content: 'Result data',
        },
      ],
    };
    const result = mapMessage(message);
    expect(result.role).toBe('tool');
  });

  it('maps assistant message with string content', () => {
    const message: Message = { role: 'assistant', content: 'Response text' };
    const result = mapMessage(message);
    expect(result.role).toBe('assistant');
    expect(result.content).toBe('Response text');
  });

  it('maps assistant message with tool use', () => {
    const message: Message = {
      role: 'assistant',
      content: [{ type: 'tool_use', id: 'call-1', name: 'test_fn', input: { key: 'val' } }],
    };
    const result = mapMessage(message);
    expect(result.role).toBe('assistant');
    expect('tool_calls' in result).toBe(true);
  });
});

// ============================================================================
// mapTool
// ============================================================================

describe('mapTool', () => {
  it('maps tool definition to OpenAI format', () => {
    const tool: ToolDefinition = {
      name: 'get_weather',
      description: 'Get weather info',
      inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
    };
    const result = mapTool(tool);
    expect(result.type).toBe('function');
    const fnResult = result as unknown as {
      type: 'function';
      function: { name: string; description: string; parameters: unknown };
    };
    expect(fnResult.function.name).toBe('get_weather');
    expect(fnResult.function.description).toBe('Get weather info');
    expect(fnResult.function.parameters).toEqual(tool.inputSchema);
  });
});

// ============================================================================
// mapResponseUsage
// ============================================================================

describe('mapResponseUsage', () => {
  it('maps usage from response', () => {
    const response = {
      usage: { prompt_tokens: 100, completion_tokens: 200, total_tokens: 300 },
    } as ChatCompletion;
    const usage = mapResponseUsage(response);
    expect(usage?.inputTokens).toBe(100);
    expect(usage?.outputTokens).toBe(200);
    expect(usage?.totalTokens).toBe(300);
  });

  it('returns undefined for missing usage rather than defaulting to 0 (#4439)', () => {
    // This previously asserted 0/0/0. That default was the bug: a synthesised
    // zero is indistinguishable downstream from a real zero-token call, and it
    // silently defeated the measured-voter gate (#4436).
    const response = {} as ChatCompletion;

    expect(mapResponseUsage(response)).toBeUndefined();
  });
});

// ============================================================================
// mapStreamChunk
// ============================================================================

describe('mapStreamChunk', () => {
  it('emits message_start on first chunk', () => {
    const chunk = {
      model: 'gpt-4',
      choices: [{ delta: { content: 'Hi' }, index: 0, finish_reason: null }],
    } as ChatCompletionChunk;
    const result = mapStreamChunk(chunk, 0, false);
    expect(result[0]?.type).toBe('message_start');
  });

  it('does not emit message_start on subsequent chunks', () => {
    const chunk = {
      model: 'gpt-4',
      choices: [{ delta: { content: 'more' }, index: 0, finish_reason: null }],
    } as ChatCompletionChunk;
    const result = mapStreamChunk(chunk, 1, true);
    expect(result.some((c) => c.type === 'message_start')).toBe(false);
  });

  it('marks the streamed prompt count as unmeasured (#4835)', () => {
    // OpenAI streaming does not report prompt_tokens on the final chunk, so
    // `inputTokens: 0` is a placeholder. Unflagged it is byte-identical to an
    // empty prompt, and a consumer billing on it prices a large-context call
    // at zero.
    const chunk = {
      model: 'gpt-4',
      choices: [{ delta: {}, index: 0, finish_reason: 'stop' }],
      usage: { completion_tokens: 42, total_tokens: 42 },
    } as unknown as ChatCompletionChunk;

    const delta = mapStreamChunk(chunk, 0, true).find((c) => c.type === 'message_delta');

    expect(delta).toBeDefined();
    expect(delta?.type === 'message_delta' ? delta.usage : undefined).toEqual(
      expect.objectContaining({ inputTokens: 0, inputTokensMeasured: false, outputTokens: 42 })
    );
  });

  it('handles empty choices', () => {
    const chunk = { model: 'gpt-4', choices: [] } as unknown as ChatCompletionChunk;
    const result = mapStreamChunk(chunk, 0, true);
    expect(result).toEqual([]);
  });

  it('emits content_block_delta for text', () => {
    const chunk = {
      model: 'gpt-4',
      choices: [{ delta: { content: 'text' }, index: 0, finish_reason: null }],
    } as ChatCompletionChunk;
    const result = mapStreamChunk(chunk, 0, true);
    expect(result.some((c) => c.type === 'content_block_delta')).toBe(true);
  });

  it('emits message_stop on finish', () => {
    const chunk = {
      model: 'gpt-4',
      choices: [{ delta: {}, index: 0, finish_reason: 'stop' }],
    } as ChatCompletionChunk;
    const result = mapStreamChunk(chunk, 0, true);
    expect(result.some((c) => c.type === 'message_stop')).toBe(true);
  });
});

// ============================================================================
// A stream whose usage was never reported must not carry a usage block
// ============================================================================

describe('mapStreamChunk usage on an unreported stream', () => {
  // OpenAI populates `chunk.usage` on a streaming response ONLY when the
  // request sets `stream_options: { include_usage: true }`, and nothing in this
  // tree does — so the `?? 0` fallbacks always took the zero branch and every
  // stream emitted `{inputTokens: 0, outputTokens: 0, totalTokens: 0}`: a usage
  // block in which nothing whatsoever was measured.
  //
  // `inputTokensMeasured: false` covered only the first of the three. Its own
  // contract is "whether `inputTokens` is a measurement", and there is no
  // `outputTokensMeasured` — so a consumer honouring the flag discounted
  // `inputTokens` and then read `outputTokens: 0` as a measured zero.
  function finalChunk(usage?: {
    completion_tokens: number;
    total_tokens: number;
  }): ChatCompletionChunk {
    return {
      model: 'gpt-4',
      choices: [{ delta: {}, index: 0, finish_reason: 'stop' }],
      ...(usage !== undefined ? { usage } : {}),
    } as ChatCompletionChunk;
  }

  it('omits usage entirely when the API reported none', () => {
    const delta = mapStreamChunk(finalChunk(), 1, true).find((c) => c.type === 'message_delta');

    expect(delta).toBeDefined();
    // The #4439 policy, stated on the field itself: where NOTHING is known,
    // omit `usage` rather than zero-fill it.
    expect(delta?.usage).toBeUndefined();
  });

  it('still emits the stop reason without usage', () => {
    // Omitting usage must not cost the chunk its actual payload.
    const delta = mapStreamChunk(finalChunk(), 1, true).find((c) => c.type === 'message_delta');

    expect(delta?.delta?.stop_reason).toBe('end_turn');
  });

  it('keeps usage when the API did report it', () => {
    // The pair. Without it, omitting usage unconditionally would pass — and
    // that would discard a real measurement once `include_usage` is wired.
    const delta = mapStreamChunk(
      finalChunk({ completion_tokens: 42, total_tokens: 42 }),
      1,
      true
    ).find((c) => c.type === 'message_delta');

    expect(delta?.usage?.outputTokens).toBe(42);
    expect(delta?.usage?.inputTokensMeasured).toBe(false);
  });
});
