from pathlib import Path


def test_miles_chat_template_compat_import_resolves_fixed_template():
    from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template

    template_path, kwargs = resolve_fixed_chat_template(TITOTokenizerType.QWEN3, ["tool"])

    assert template_path is not None
    assert Path(template_path).name == "qwen3_fixed.jinja"
    assert Path(template_path).is_file()
    assert kwargs == {}


def test_miles_session_compat_import_resolves_errors():
    from miles.rollout.session.session_errors import SessionError, SessionNotFoundError

    assert SessionError.status_code == 500
    assert SessionNotFoundError.status_code == 404


def test_public_get_tito_tokenizer_delegates_to_vendored_default():
    from tito_gateway import get_tito_tokenizer
    from tito_gateway.vendor.miles_compat.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

    fake_tokenizer = object()

    result = get_tito_tokenizer(fake_tokenizer, tokenizer_type="default")

    assert isinstance(result, TITOTokenizer)
    assert result.tokenizer is fake_tokenizer
