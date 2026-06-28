"""CLI implementation for Miles chat-template append-only verification."""

from __future__ import annotations

from typing import Any


def run_from_args(args: Any) -> int:
    """Run the vendored Miles chat-template verifier from an argparse namespace."""
    if args.model is None and args.template is None:
        raise ValueError("one of --model or --template is required")
    if args.tito_model is not None and args.model is None:
        raise ValueError("--tito-model requires --model so the TITO verifier can load the tokenizer")

    extra_template_kwargs = dict(args.chat_template_kwargs or {})
    allowed_roles = set(args.tito_allowed_append_roles) | {"tool"}
    use_tito_instance = args.tito_model is not None

    if use_tito_instance:
        from miles.utils.chat_template_utils import resolve_fixed_chat_template
        from miles.utils.processing_utils import load_tokenizer

        fixed_path, resolved_kwargs = resolve_fixed_chat_template(args.tito_model, sorted(allowed_roles))
        for key, value in resolved_kwargs.items():
            if key in extra_template_kwargs:
                continue
            extra_template_kwargs[key] = value
            print(f"Auto-set --chat-template-kwargs {key}={value!r} (from --tito-model={args.tito_model})")

        template_path = args.template or fixed_path
        tokenizer = load_tokenizer(args.model, chat_template_path=template_path, trust_remote_code=True)
        if args.template:
            source_desc = f"template override via TITO: {args.template}"
        elif fixed_path:
            source_desc = f"fixed template via TITO: {fixed_path}"
        elif getattr(tokenizer, "chat_template", None) is not None:
            source_desc = f"HuggingFace via TITO: {args.model}"
        else:
            source_desc = f"TITO encoder: {args.tito_model}"
        chat_template = None
    elif args.template:
        with open(args.template) as f:
            chat_template = f.read()
        source_desc = f"file: {args.template}"
        tokenizer = None
    else:
        from miles.utils.chat_template_utils.template import load_hf_chat_template

        chat_template = load_hf_chat_template(args.model)
        source_desc = f"HuggingFace: {args.model}"
        tokenizer = None

    from miles.utils.test_utils.chat_template_verify import (
        ALL_CASES,
        check_coverage,
        run_all_checks,
        run_all_checks_via_tito,
        select_cases,
    )

    is_thinking_filter = {"off": False, "on": True, "both": None}[args.thinking]
    selected = select_cases(allowed_append_roles=allowed_roles, is_thinking=is_thinking_filter)

    print(f"Template source:       {source_desc}")
    print(f"Allowed append roles:  {sorted(allowed_roles)}")
    print(f"Thinking mode:         {args.thinking}")
    if extra_template_kwargs:
        print(f"Template kwargs:       {extra_template_kwargs}")
    print(f"Selected trajectories: {len(selected)} of {len(ALL_CASES)} (after filtering)")
    print()

    coverage = check_coverage()
    if coverage.missing:
        print("Trajectory coverage gaps ((thinking, append_roles \\ {tool}) with no trajectory):")
        for is_thinking, roles in coverage.missing:
            label = "thinking    " if is_thinking else "non-thinking"
            roles_str = "{" + ", ".join(roles) + "}" if roles else "{}"
            print(f"  - {label}  x  {roles_str}")
        print()

    if use_tito_instance:
        results = run_all_checks_via_tito(
            tokenizer,
            args.tito_model,
            allowed_append_roles=allowed_roles,
            thinking=args.thinking,
            extra_template_kwargs=extra_template_kwargs,
        )
    else:
        results = run_all_checks(
            chat_template,
            allowed_append_roles=allowed_roles,
            thinking=args.thinking,
            extra_template_kwargs=extra_template_kwargs,
        )

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    max_name_len = max((len(r.case_name) for r in results), default=0)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = f"  [{status}] {r.case_name:<{max_name_len}}"
        if r.error:
            first_line = r.error.split("\n")[0]
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            line += f"  -- {first_line}"
        print(line)

    print()
    print(f"Results: {passed}/{len(results)} passed, {failed} failed")

    if failed:
        if use_tito_instance:
            print("\nVerdict: FAIL - TITO incremental tokenization did NOT match standard render")
        else:
            print("\nVerdict: FAIL - template is NOT append-only after last user message")
        return 1

    if use_tito_instance:
        print("\nVerdict: PASS - TITO incremental tokenization matched standard render")
    else:
        print("\nVerdict: PASS - template IS append-only after last user message")
    return 0
