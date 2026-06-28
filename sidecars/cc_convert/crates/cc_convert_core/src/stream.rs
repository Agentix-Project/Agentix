//! OpenAI SSE chunk stream → Anthropic SSE event stream.
//!
//! Push chunks one at a time via [`StreamTranslator::push_openai_chunk`].
//! Call [`StreamTranslator::finish`] when the upstream closes; it emits the
//! `message_delta` + `message_stop` events if they were not already emitted
//! due to a `finish_reason`-bearing chunk.

use crate::anthropic::*;
use crate::openai::*;
use crate::resp_to_anthropic::{map_stop_reason, map_usage, rewrite_id};
use crate::tool_names::ToolNameMap;
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct StreamConvertOptions {
    /// Anthropic SDK convention is to emit a `ping` event right after
    /// `message_start`. LiteLLM does NOT emit it; default false for parity.
    pub emit_ping: bool,
    /// Anthropic spec includes `stop_sequence: null` in `message_delta`.
    /// LiteLLM omits it; default false for parity.
    pub include_stop_sequence_in_message_delta: bool,
    /// `message_start.message.usage` always includes
    /// `cache_creation_input_tokens` and `cache_read_input_tokens` (LiteLLM
    /// behaviour). Default true.
    pub include_zero_cache_fields_in_usage: bool,
    /// Generate a fresh `msg_<uuid>` for the message id (LiteLLM behaviour).
    /// Default false: derive from the OpenAI `chatcmpl-*` id instead.
    pub random_message_id: bool,
    /// LiteLLM eagerly opens the first `content_block_start` (text block at
    /// index 0) immediately after `message_start`, even before any delta
    /// arrives. Default true for parity.
    pub eager_open_text_block: bool,
}

impl Default for StreamConvertOptions {
    fn default() -> Self {
        Self {
            emit_ping: false,
            include_stop_sequence_in_message_delta: false,
            include_zero_cache_fields_in_usage: true,
            random_message_id: false,
            eager_open_text_block: true,
        }
    }
}

impl StreamConvertOptions {
    pub fn litellm_compat() -> Self {
        Self::default()
    }

    /// Matches the Anthropic SDK's published SSE shape (with ping +
    /// stop_sequence + lazy block opening).
    pub fn anthropic_native() -> Self {
        Self {
            emit_ping: true,
            include_stop_sequence_in_message_delta: true,
            include_zero_cache_fields_in_usage: false,
            random_message_id: false,
            eager_open_text_block: false,
        }
    }
}

#[derive(Debug, Clone)]
struct ToolBlockState {
    anthropic_index: i32,
    started: bool,
    /// Remembered from the first chunk; continuation chunks may not carry it.
    id: String,
    /// Remembered from the first chunk; continuation chunks may not carry it.
    name: String,
}

#[derive(Debug)]
pub struct StreamTranslator {
    original_model: String,
    tool_names: ToolNameMap,
    opts: StreamConvertOptions,

    sent_message_start: bool,
    text_block_open: bool,
    text_block_index: i32,

    thinking_block_open: bool,
    thinking_block_index: i32,

    tool_blocks: BTreeMap<i32, ToolBlockState>,
    next_anthropic_index: i32,

    pending_usage: Option<OpenAIUsage>,
    pending_stop_reason: Option<AnthropicStopReason>,
    emitted_stop: bool,
}

impl StreamTranslator {
    pub fn new(original_model: String, tool_names: ToolNameMap) -> Self {
        Self::with_options(original_model, tool_names, StreamConvertOptions::default())
    }

    pub fn with_options(
        original_model: String,
        tool_names: ToolNameMap,
        opts: StreamConvertOptions,
    ) -> Self {
        Self {
            original_model,
            tool_names,
            opts,
            sent_message_start: false,
            text_block_open: false,
            text_block_index: 0,
            thinking_block_open: false,
            thinking_block_index: 0,
            tool_blocks: BTreeMap::new(),
            next_anthropic_index: 0,
            pending_usage: None,
            pending_stop_reason: None,
            emitted_stop: false,
        }
    }

    pub fn push_openai_chunk(&mut self, chunk: &OpenAIStreamChunk) -> Vec<AnthropicEvent> {
        let mut out = Vec::new();
        if self.emitted_stop {
            return out;
        }

        if !self.sent_message_start {
            self.send_message_start(chunk, &mut out);
        }

        if let Some(usage) = &chunk.usage {
            self.pending_usage = Some(usage.clone());
        }

        for choice in &chunk.choices {
            self.process_choice(choice, &mut out);
        }

        out
    }

    fn send_message_start(&mut self, chunk: &OpenAIStreamChunk, out: &mut Vec<AnthropicEvent>) {
        let id = if self.opts.random_message_id {
            format!("msg_{}", uuid::Uuid::new_v4())
        } else {
            chunk
                .id
                .as_deref()
                .map(rewrite_id)
                .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4()))
        };
        let usage = if self.opts.include_zero_cache_fields_in_usage {
            AnthropicUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: Some(0),
                cache_read_input_tokens: Some(0),
            }
        } else {
            AnthropicUsage::default()
        };
        out.push(AnthropicEvent::MessageStart {
            message: MessageStartPayload {
                id,
                msg_type: "message".to_string(),
                role: "assistant".to_string(),
                model: self.original_model.clone(),
                content: Vec::new(),
                stop_reason: None,
                stop_sequence: None,
                usage,
            },
        });
        if self.opts.emit_ping {
            out.push(AnthropicEvent::Ping);
        }
        if self.opts.eager_open_text_block {
            // Open content_block index 0 as a text block eagerly. Tool calls
            // arriving later will allocate their own indices.
            let idx = self.allocate_index();
            self.text_block_index = idx;
            self.text_block_open = true;
            out.push(AnthropicEvent::ContentBlockStart {
                index: idx,
                content_block: StreamingContentBlock::Text { text: String::new() },
            });
        }
        self.sent_message_start = true;
    }

    fn process_choice(&mut self, choice: &OpenAIStreamChoice, out: &mut Vec<AnthropicEvent>) {
        let delta = &choice.delta;

        let reasoning = delta
            .reasoning_content
            .as_deref()
            .or(delta.reasoning.as_deref());
        if let Some(t) = reasoning.filter(|s| !s.is_empty()) {
            if !self.thinking_block_open {
                // Close text block first if eagerly opened but empty.
                if self.text_block_open {
                    out.push(AnthropicEvent::ContentBlockStop {
                        index: self.text_block_index,
                    });
                    self.text_block_open = false;
                }
                let idx = self.allocate_index();
                self.thinking_block_index = idx;
                out.push(AnthropicEvent::ContentBlockStart {
                    index: idx,
                    content_block: StreamingContentBlock::Thinking {
                        thinking: String::new(),
                    },
                });
                self.thinking_block_open = true;
            }
            out.push(AnthropicEvent::ContentBlockDelta {
                index: self.thinking_block_index,
                delta: BlockDelta::ThinkingDelta {
                    thinking: t.to_string(),
                },
            });
        }

        if let Some(text) = delta.content.as_deref().filter(|s| !s.is_empty()) {
            if !self.text_block_open {
                if self.thinking_block_open {
                    out.push(AnthropicEvent::ContentBlockStop {
                        index: self.thinking_block_index,
                    });
                    self.thinking_block_open = false;
                }
                let idx = self.allocate_index();
                self.text_block_index = idx;
                out.push(AnthropicEvent::ContentBlockStart {
                    index: idx,
                    content_block: StreamingContentBlock::Text {
                        text: String::new(),
                    },
                });
                self.text_block_open = true;
            }
            out.push(AnthropicEvent::ContentBlockDelta {
                index: self.text_block_index,
                delta: BlockDelta::TextDelta {
                    text: text.to_string(),
                },
            });
        }

        if let Some(tcs) = delta.tool_calls.as_ref() {
            if self.text_block_open {
                out.push(AnthropicEvent::ContentBlockStop {
                    index: self.text_block_index,
                });
                self.text_block_open = false;
            }
            if self.thinking_block_open {
                out.push(AnthropicEvent::ContentBlockStop {
                    index: self.thinking_block_index,
                });
                self.thinking_block_open = false;
            }

            for tc in tcs {
                let oi = tc.index.unwrap_or(0);
                // Some upstreams (vLLM continuations, SGLang null-id) omit
                // id/name on continuation chunks. Use the chunk's values if
                // present, otherwise fall back to what we recorded on the
                // first chunk for this index.
                let incoming_id = tc.id.clone();
                let incoming_name = tc.function.name.clone();
                let state = self.tool_blocks.entry(oi).or_insert(ToolBlockState {
                    anthropic_index: self.next_anthropic_index,
                    started: false,
                    id: incoming_id.clone().unwrap_or_default(),
                    name: incoming_name.clone().unwrap_or_default(),
                });
                // Update remembered id/name if this chunk provided them and
                // we didn't have them before.
                if state.id.is_empty() {
                    if let Some(id) = incoming_id {
                        state.id = id;
                    }
                }
                if state.name.is_empty() {
                    if let Some(n) = incoming_name {
                        state.name = n;
                    }
                }
                let ai = state.anthropic_index;
                if ai == self.next_anthropic_index {
                    // New block — bump the counter (entry().or_insert reserved it).
                    self.next_anthropic_index += 1;
                }
                if !state.started && !state.id.is_empty() {
                    // We've seen enough to open the block.
                    let restored = self.tool_names.restore(&state.name).to_string();
                    let id_owned = state.id.clone();
                    out.push(AnthropicEvent::ContentBlockStart {
                        index: ai,
                        content_block: StreamingContentBlock::ToolUse {
                            id: id_owned,
                            name: restored,
                            input: json!({}),
                        },
                    });
                    // Re-borrow to flip started since the immutable borrow is done.
                    self.tool_blocks.get_mut(&oi).unwrap().started = true;
                }
                if let Some(args) = tc.function.arguments.as_deref().filter(|s| !s.is_empty()) {
                    out.push(AnthropicEvent::ContentBlockDelta {
                        index: ai,
                        delta: BlockDelta::InputJsonDelta {
                            partial_json: args.to_string(),
                        },
                    });
                }
            }
        }

        if let Some(reason) = choice.finish_reason.as_deref() {
            self.pending_stop_reason = Some(map_stop_reason(Some(reason)));
            self.emit_close(out);
        }
    }

    fn allocate_index(&mut self) -> i32 {
        let i = self.next_anthropic_index;
        self.next_anthropic_index += 1;
        i
    }

    fn emit_close(&mut self, out: &mut Vec<AnthropicEvent>) {
        if self.emitted_stop {
            return;
        }
        if self.text_block_open {
            out.push(AnthropicEvent::ContentBlockStop {
                index: self.text_block_index,
            });
            self.text_block_open = false;
        }
        if self.thinking_block_open {
            out.push(AnthropicEvent::ContentBlockStop {
                index: self.thinking_block_index,
            });
            self.thinking_block_open = false;
        }
        for (_, state) in self.tool_blocks.iter_mut() {
            if state.started {
                out.push(AnthropicEvent::ContentBlockStop {
                    index: state.anthropic_index,
                });
                state.started = false;
            }
        }

        let usage = map_usage(self.pending_usage.as_ref(), true);
        let stop_reason = self
            .pending_stop_reason
            .take()
            .unwrap_or(AnthropicStopReason::EndTurn);

        out.push(AnthropicEvent::MessageDelta {
            delta: MessageDeltaPayload {
                stop_reason: Some(stop_reason),
                stop_sequence: if self.opts.include_stop_sequence_in_message_delta {
                    None
                } else {
                    None
                },
            },
            usage,
        });
        out.push(AnthropicEvent::MessageStop);
        self.emitted_stop = true;
    }

    pub fn finish(&mut self) -> Vec<AnthropicEvent> {
        let mut out = Vec::new();
        if !self.sent_message_start {
            self.send_message_start(
                &OpenAIStreamChunk {
                    id: None,
                    model: None,
                    choices: Vec::new(),
                    usage: None,
                },
                &mut out,
            );
        }
        self.emit_close(&mut out);
        out
    }
}
