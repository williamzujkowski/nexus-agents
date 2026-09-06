/**
 * nexus-agents/adapters - OpenAI Message Mappers
 *
 * Functions for mapping between Nexus and OpenAI message formats.
 */

import type OpenAI from 'openai';
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/completions';
import type {
  ContentBlock,
  Message,
  ToolDefinition,
  TokenUsage,
  StopReason,
  StreamChunk,
} from '../core/index.js';
import { isFunctionToolCall } from './openai-types.js';

/**
 * Maps OpenAI finish reasons to our StopReason type.
 */
export function mapStopReason(openaiReason: string | null | undefined): StopReason {
  switch (openaiReason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'end_turn';
    default:
      return 'end_turn';
  }
}

/**
 * Maps OpenAI choice to our ContentBlock array.
 */
export function mapChoiceToContentBlocks(choice: ChatCompletion.Choice): ContentBlock[] {
  const blocks: ContentBlock[] = [];
  const message = choice.message;

  // Add text content if present
  if (message.content !== null && message.content !== '') {
    blocks.push({ type: 'text', text: message.content });
  }

  // Add tool calls if present
  if (message.tool_calls !== undefined && message.tool_calls.length > 0) {
    for (const toolCall of message.tool_calls) {
      if (isFunctionToolCall(toolCall)) {
        let parsedInput: unknown;
        try {
          parsedInput = JSON.parse(toolCall.function.arguments) as unknown;
        } catch {
          parsedInput = { _raw: toolCall.function.arguments };
        }
        blocks.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.function.name,
          input: parsedInput,
        });
      }
    }
  }

  // If no content at all, return empty text block
  if (blocks.length === 0) {
    blocks.push({ type: 'text', text: '' });
  }

  return blocks;
}

/**
 * Maps our Message format to OpenAI's ChatCompletionMessageParam format.
 */
export function mapMessage(message: Message): ChatCompletionMessageParam {
  // Handle system messages
  if (message.role === 'system') {
    return mapSystemMessage(message);
  }

  // Handle user messages
  if (message.role === 'user') {
    return mapUserMessage(message);
  }

  // Handle assistant messages
  return mapAssistantMessage(message);
}

/**
 * Maps a system message.
 */
function mapSystemMessage(message: Message): ChatCompletionMessageParam {
  const content =
    typeof message.content === 'string'
      ? message.content
      : message.content
          .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
          .map((b) => b.text)
          .join('\n');
  return { role: 'system', content };
}

/**
 * Maps a user message.
 */
function mapUserMessage(message: Message): ChatCompletionMessageParam {
  if (typeof message.content === 'string') {
    return { role: 'user', content: message.content };
  }

  // Check for tool results - these need special handling
  const toolResults = message.content.filter(
    (b): b is { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean } =>
      b.type === 'tool_result'
  );

  if (toolResults.length > 0) {
    // Return first tool result as a tool message
    const firstResult = toolResults[0];
    if (firstResult !== undefined) {
      return {
        role: 'tool',
        tool_call_id: firstResult.tool_use_id,
        content: firstResult.content,
      };
    }
  }

  // Map to user message with content array
  const content = message.content.map((block) => {
    if (block.type === 'text') {
      return { type: 'text' as const, text: block.text };
    }
    if (block.type === 'image') {
      return {
        type: 'image_url' as const,
        image_url: {
          url: `data:${block.source.media_type};base64,${block.source.data}`,
        },
      };
    }
    // Fallback for other types
    return { type: 'text' as const, text: '' };
  });

  return { role: 'user', content };
}

/**
 * Maps an assistant message.
 */
function mapAssistantMessage(message: Message): ChatCompletionMessageParam {
  if (typeof message.content === 'string') {
    return { role: 'assistant', content: message.content };
  }

  // Check for tool uses in assistant message
  const toolUses = message.content.filter(
    (b): b is { type: 'tool_use'; id: string; name: string; input: unknown } =>
      b.type === 'tool_use'
  );

  const textContent = message.content
    .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
    .map((b) => b.text)
    .join('');

  if (toolUses.length > 0) {
    const assistantMessage: ChatCompletionMessageParam = {
      role: 'assistant',
      content: textContent !== '' ? textContent : null,
      tool_calls: toolUses.map((tool) => ({
        id: tool.id,
        type: 'function' as const,
        function: {
          name: tool.name,
          arguments: JSON.stringify(tool.input),
        },
      })),
    };
    return assistantMessage;
  }

  return { role: 'assistant', content: textContent };
}

/**
 * Maps our ToolDefinition to OpenAI's tool format.
 */
export function mapTool(tool: ToolDefinition): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  };
}

/**
 * Maps OpenAI API response to our CompletionResponse format.
 */
export function mapResponseUsage(response: ChatCompletion): TokenUsage | undefined {
  const u = response.usage;
  // #4439: `?? 0` here synthesised a measurement the vendor never sent. An
  // absent usage block must stay absent so the decision-cost rollup can tell
  // "zero tokens" from "we do not know".
  if (u === undefined) return undefined;
  const cached = u.prompt_tokens_details?.cached_tokens;
  return {
    inputTokens: u.prompt_tokens,
    outputTokens: u.completion_tokens,
    totalTokens: u.total_tokens,
    ...(cached !== undefined ? { cachedInputTokens: cached } : {}),
  };
}

/**
 * Maps content delta from stream chunk.
 */
function mapContentDelta(
  delta: OpenAI.Chat.ChatCompletionChunk.Choice.Delta,
  currentIndex: number,
  hasStarted: boolean
): StreamChunk[] {
  const chunks: StreamChunk[] = [];

  if (delta.content !== undefined && delta.content !== null && delta.content !== '') {
    // Start content block if this is the first content
    if (currentIndex === 0 && !hasStarted) {
      chunks.push({
        type: 'content_block_start',
        index: 0,
        contentBlock: { type: 'text', text: '' },
      });
    }

    chunks.push({
      type: 'content_block_delta',
      index: currentIndex,
      delta: { type: 'text_delta', text: delta.content },
    });
  }

  return chunks;
}

/**
 * Maps tool calls from stream chunk delta.
 */
function mapToolCallsDelta(delta: OpenAI.Chat.ChatCompletionChunk.Choice.Delta): StreamChunk[] {
  const chunks: StreamChunk[] = [];

  if (delta.tool_calls !== undefined && delta.tool_calls.length > 0) {
    for (const toolCall of delta.tool_calls) {
      if (toolCall.function?.name !== undefined) {
        chunks.push({
          type: 'content_block_start',
          index: toolCall.index,
          contentBlock: {
            type: 'tool_use',
            id: toolCall.id ?? '',
            name: toolCall.function.name,
            input: {},
          },
        });
      }
    }
  }

  return chunks;
}

/**
 * Maps finish reason from stream chunk.
 */
function mapFinishChunks(
  choice: ChatCompletionChunk.Choice,
  chunk: ChatCompletionChunk,
  currentIndex: number
): StreamChunk[] {
  const chunks: StreamChunk[] = [];

  if (choice.finish_reason !== null) {
    // End current content block
    chunks.push({
      type: 'content_block_stop',
      index: currentIndex,
    });

    // Emit message_delta with stop reason.
    //
    // NO `usage` on this path. OpenAI populates `chunk.usage` on a streaming
    // response only when the request sets `stream_options: { include_usage:
    // true }`, and nothing in this tree does — so both `??` fallbacks always
    // took the `0` branch and every stream emitted
    // `{inputTokens: 0, outputTokens: 0, totalTokens: 0}`: a usage block in
    // which nothing whatsoever was measured.
    //
    // `inputTokensMeasured: false` covered only the first of those three. Its
    // own contract is "whether `inputTokens` is a measurement", and there is no
    // `outputTokensMeasured`, so a consumer honouring the flag correctly
    // discounted `inputTokens` and then read `outputTokens: 0` as a measured
    // zero. That is the #4439 policy this violated, stated on the field itself:
    // "Where NOTHING is known, prefer omitting `usage` entirely." The SDK
    // adapter's stream path already does exactly that (#4835), for the same
    // reason.
    chunks.push({
      type: 'message_delta',
      delta: { stop_reason: mapStopReason(choice.finish_reason) },
      // A stream whose usage IS reported keeps it — when `include_usage` is
      // wired, this is the branch that carries a real measurement.
      ...(chunk.usage !== undefined && chunk.usage !== null
        ? {
            usage: {
              // OpenAI still omits prompt_tokens on the final chunk (#4835).
              inputTokens: 0,
              inputTokensMeasured: false,
              outputTokens: chunk.usage.completion_tokens,
              totalTokens: chunk.usage.total_tokens,
            },
          }
        : {}),
    });

    // Emit message_stop
    chunks.push({ type: 'message_stop' });
  }

  return chunks;
}

/**
 * Maps OpenAI stream chunks to our StreamChunk format.
 */
export function mapStreamChunk(
  chunk: ChatCompletionChunk,
  currentIndex: number,
  hasStarted: boolean
): StreamChunk[] {
  const chunks: StreamChunk[] = [];
  const choice = chunk.choices[0];

  // Emit message_start on first chunk
  if (!hasStarted) {
    chunks.push({
      type: 'message_start',
      message: { model: chunk.model },
    });
  }

  if (choice === undefined) {
    return chunks;
  }

  const delta = choice.delta;

  // Map content delta
  chunks.push(...mapContentDelta(delta, currentIndex, hasStarted));

  // Map tool calls
  chunks.push(...mapToolCallsDelta(delta));

  // Map finish reason
  chunks.push(...mapFinishChunks(choice, chunk, currentIndex));

  return chunks;
}
