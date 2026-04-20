ACTIONS = ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"]


def test_parses_simple_action():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    assert parse_action_cot("ACTION: MOVE_LEFT", ACTIONS) == 0
    assert parse_action_cot("ACTION: MOVE_RIGHT", ACTIONS) == 1
    assert parse_action_cot("ACTION: ATTACK", ACTIONS) == 2


def test_parses_with_preceding_think():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    text = "THOUGHTS: monster on the right, shoot\nACTION: ATTACK"
    assert parse_action_cot(text, ACTIONS) == 2


def test_takes_last_match_on_repeated_action_pathology():
    """Pathology: the model emits multiple ACTION lines. We take the last."""
    from cleanrl_vlm.prompts.parser import parse_action_cot

    text = "ACTION: MOVE_LEFT\nSorry, actually:\nACTION: ATTACK"
    assert parse_action_cot(text, ACTIONS) == 2


def test_whitelist_rejects_unknown_action():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    assert parse_action_cot("ACTION: RUN_AWAY", ACTIONS) is None


def test_returns_none_on_no_match():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    assert parse_action_cot("blah blah no tag", ACTIONS) is None


def test_is_case_sensitive_but_strips_trailing_whitespace():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    assert parse_action_cot("ACTION: MOVE_LEFT  \n\n", ACTIONS) == 0


def test_regex_tolerates_extra_spaces_after_colon():
    from cleanrl_vlm.prompts.parser import parse_action_cot

    assert parse_action_cot("ACTION:    ATTACK", ACTIONS) == 2
