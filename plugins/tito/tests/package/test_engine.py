"""Self-contained tests for the native TITO engine.

These build a tiny in-memory tokenizer (no model download) and assert the engine's
invariants directly: the incremental tokenization equals a from-scratch render, the
comparator classifies mismatches correctly, message matching collapses falsy
sentinels, and the session state machine rolls back to the last assistant checkpoint.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from agentix.tito.engine.compare import MismatchType, TokenSeqComparator
from agentix.tito.engine.messages import assert_messages_append_only_with_allowed_role, message_matches
from agentix.tito.engine.pretokenize import Qwen3TITOTokenizer, get_tito_tokenizer
from agentix.tito.engine.trajectory import LinearTrajectory, SessionRegistry


@pytest.fixture(scope="module")
def tok():
    specials = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    words = ["system", "user", "assistant", "tool", "dummy", "You", "are", "ok",
             "done", "compute", "17", "23", "391", "X", "Y", "Hello"]
    vocab = {t: i for i, t in enumerate(specials + words)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    t.chat_template = (
        "{%- for m in messages -%}<|im_start|>{{ m['role'] }} {{ m['content'] or '' }}<|im_end|>{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|im_start|>assistant {%- endif -%}"
    )
    return t


def _types(ms):
    return [(m.type, m.segment_index) for m in ms]


def test_comparator_classifies_mismatches(tok):
    cmp = TokenSeqComparator(tok, assistant_start_str="<|im_start|>assistant")
    ims, ime = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    S, U, A = (tok.convert_tokens_to_ids(w) for w in ("system", "user", "assistant"))
    Y391, Y23, Yok, YH = (tok.convert_tokens_to_ids(w) for w in ("391", "23", "ok", "Hello"))

    assert cmp.compare_sequences([ims, U, Y391, ime], [ims, U, Y391, ime]) == []
    assert _types(cmp.compare_sequences([ims, U, Y391, ime], [ims, U, Y23, ime])) == [
        (MismatchType.NON_ASSISTANT_TEXT, 1)
    ]
    assert _types(cmp.compare_sequences([ims, A, Yok, ime], [ims, A, YH, ime])) == [
        (MismatchType.ASSISTANT_TEXT, 1)
    ]
    assert _types(cmp.compare_sequences([ims, Y391, ime], [ims, Y391, ime, ims])) == [
        (MismatchType.SPECIAL_TOKEN_COUNT, -1)
    ]
    assert _types(cmp.compare_sequences([ims, Y391, ime], [ime, Y391, ims])) == [
        (MismatchType.SPECIAL_TOKEN_TYPE, 0),
        (MismatchType.SPECIAL_TOKEN_TYPE, 2),
    ]
    # trailing trim removes false structural diffs
    assert cmp.compare_sequences([ims, Y391, ime], [ims, Y391, ime, ime], trim_trailing_ids={ime}) == []


def test_message_matches_collapses_falsy_sentinels():
    assert message_matches({"role": "a", "content": ""}, {"role": "a", "content": None})
    assert message_matches({"role": "a", "tool_calls": []}, {"role": "a", "tool_calls": None})
    assert message_matches({"role": "u", "content": "x"}, {"role": "u", "content": "x", "extra": 1})
    # reasoning_content "\n\n" is non-falsy → not collapsed (the bug we hit)
    assert not message_matches({"role": "a", "reasoning_content": "\n\n"}, {"role": "a", "reasoning_content": None})
    assert not message_matches({"role": "u", "content": "x"}, {"role": "t", "content": "x"})


def test_append_only_enforced():
    stored = [{"role": "user", "content": "x"}]
    assert_messages_append_only_with_allowed_role(stored, stored + [{"role": "tool", "content": "y"}], ["tool"])
    with pytest.raises(ValueError):
        assert_messages_append_only_with_allowed_role(stored, stored + [{"role": "user", "content": "z"}], ["tool"])
    with pytest.raises(ValueError):
        assert_messages_append_only_with_allowed_role(stored, [{"role": "user", "content": "DIFF"}], ["tool"])


@pytest.mark.parametrize(
    "appends",
    [
        [{"role": "tool", "content": "391"}],
        [{"role": "tool", "content": "391"}, {"role": "tool", "content": "23"}],
        [{"role": "user", "content": "Hello"}],
        [{"role": "tool", "content": "X"}, {"role": "user", "content": "Y"}],
    ],
)
def test_incremental_equals_full_render(tok, appends):
    """The core invariant: merge(prefix, incremental) == full from-scratch render."""
    tt = get_tito_tokenizer(tok, "default", allowed_append_roles=("tool", "user"))
    old = [{"role": "system", "content": "You are"}, {"role": "user", "content": "compute 17 23"},
           {"role": "assistant", "content": "ok"}]
    new = old + appends
    prefix = tt.render_messages(old, add_generation_prompt=False, tokenize=True)
    merged = tt.merge_tokens(old, new, prefix, None)
    full = tt.render_messages(new, add_generation_prompt=True, tokenize=True)
    assert merged == full


def test_qwen3_newline_fixup():
    class FakeTok:
        def encode(self, t, add_special_tokens=False):
            return [99]  # "\n" -> single id

        def convert_tokens_to_ids(self, t):
            return 88  # "<|im_end|>"

    q = Qwen3TITOTokenizer(FakeTok(), chat_template_kwargs={"chat_template": "x"})
    q.tokenize_additional_non_assistant = lambda o, n, t=None: [1, 2, 3]
    assert q.merge_tokens([], [], [7, 88], None) == [7, 88, 99, 1, 2, 3]  # prefix ends in im_end -> insert \n
    assert q.merge_tokens([], [], [7, 5], None) == [7, 5, 1, 2, 3]  # otherwise no insert


def test_trajectory_rollback_to_assistant_checkpoint(tok):
    tt = get_tito_tokenizer(tok, "default", allowed_append_roles=("tool", "user"))
    reg = SessionRegistry(None, tok, tito_tokenizer=tt)
    tr = LinearTrajectory()
    sys = [{"role": "system", "content": "You are"}, {"role": "user", "content": "compute 17 23"}]
    a0 = {"role": "assistant", "content": "ok"}

    tr.prepare_pretokenized(sys, None, tito_tokenizer=tt)
    tr.update_pretokenized_state(sys, a0, tt.render_messages(sys + [a0], add_generation_prompt=False, tokenize=True), [], tt.max_trim_tokens)

    m1 = sys + [a0, {"role": "tool", "content": "391"}]
    a1 = {"role": "assistant", "content": "done"}
    tr.prepare_pretokenized(m1, None, tito_tokenizer=tt)
    tr.update_pretokenized_state(m1, a1, tt.render_messages(m1 + [a1], add_generation_prompt=False, tokenize=True), [], tt.max_trim_tokens)
    assert tr.num_assistant == 2
    assert reg.compute_session_mismatch(tr) == []  # clean chain → no mismatch

    # retry the tool turn with a different result → rollback to a0 checkpoint
    tr.prepare_pretokenized(sys + [a0, {"role": "tool", "content": "X"}], None, tito_tokenizer=tt)
    assert tr.num_assistant == 1
    assert [m.get("role") for m in tr.messages] == ["system", "user", "assistant"]
