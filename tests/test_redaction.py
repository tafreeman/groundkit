"""Redactor / RedactionPattern / RedactionConfig tests (Phase 5, SPEC.md §2).

Bug-catcher weighted per the design brief: overlap tie-break, token
stability within and across calls, per-category counters, construction-time
rejection of a non-compiling or zero-width pattern, duplicate pattern names,
restore() round-trip and its two fail-closed/pass-through branches, unicode
text, and the documented token-format collision.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundkit.providers.redaction import (
    RedactionConfig,
    RedactionPattern,
    Redactor,
    UnknownRedactionTokenError,
)


class TestRedactionPatternValidation:
    def test_valid_pattern_constructs(self) -> None:
        pattern = RedactionPattern(name="EMAIL", regex=r"[a-z]+@[a-z]+")
        assert pattern.token_category == "EMAIL"  # noqa: S105 — token category, not a credential

    def test_token_category_uppercases_name(self) -> None:
        pattern = RedactionPattern(name="Email_2", regex=r"x+")
        assert pattern.token_category == "EMAIL_2"  # noqa: S105 — token category, not a credential

    def test_non_compiling_regex_rejected(self) -> None:
        with pytest.raises(ValidationError, match="does not compile"):
            RedactionPattern(name="BAD", regex=r"[unclosed")

    @pytest.mark.parametrize(
        "regex",
        ["a*", ".*", "(foo)?", "x{0,3}", "", "foo|"],
        ids=[
            "star",
            "dot-star",
            "optional-group",
            "bounded-optional",
            "empty-literal",
            "alt-empty",
        ],
    )
    def test_pattern_matching_empty_string_rejected(self, regex: str) -> None:
        with pytest.raises(ValidationError, match="empty string"):
            RedactionPattern(name="BAD", regex=regex)

    def test_pattern_requiring_at_least_one_char_accepted(self) -> None:
        # Regression guard for the empty-string check above: it must not be
        # so aggressive that it rejects an ordinary "one or more" pattern.
        RedactionPattern(name="OK", regex=r"a+")

    @pytest.mark.parametrize(
        "name",
        ["", "1EMAIL", "_EMAIL", "EMAIL NAME", "EMAIL-NAME", "EMAIL.NAME", "émail"],
        ids=["empty", "leading-digit", "leading-underscore", "space", "hyphen", "dot", "unicode"],
    )
    def test_invalid_name_rejected(self, name: str) -> None:
        with pytest.raises(ValidationError):
            RedactionPattern(name=name, regex=r"x+")

    @pytest.mark.parametrize("name", ["email", "EMAIL", "Email_2", "a", "A1_b2"])
    def test_valid_name_variants_accepted(self, name: str) -> None:
        RedactionPattern(name=name, regex=r"x+")


class TestRedactionConfigValidation:
    def test_default_config_has_expected_categories(self) -> None:
        categories = {p.token_category for p in RedactionConfig().patterns}
        assert categories == {"EMAIL", "PHONE_E164", "PHONE_US", "IPV4", "SECRET_TOKEN"}

    def test_duplicate_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            RedactionConfig(
                patterns=(
                    RedactionPattern(name="EMAIL", regex=r"x+"),
                    RedactionPattern(name="EMAIL", regex=r"y+"),
                )
            )

    def test_duplicate_name_case_insensitive_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            RedactionConfig(
                patterns=(
                    RedactionPattern(name="Email", regex=r"x+"),
                    RedactionPattern(name="EMAIL", regex=r"y+"),
                )
            )

    def test_distinct_names_accepted(self) -> None:
        config = RedactionConfig(
            patterns=(
                RedactionPattern(name="ALPHA", regex=r"x+"),
                RedactionPattern(name="BETA", regex=r"y+"),
            )
        )
        assert len(config.patterns) == 2

    def test_empty_pattern_set_accepted_as_a_no_op_config(self) -> None:
        config = RedactionConfig(patterns=())
        redactor = Redactor(config)
        result = redactor.redact("nothing here should change")
        assert result.text == "nothing here should change"
        assert result.mapping == ()


class TestOverlapTieBreak:
    """Leftmost-longest, deterministic — the exact rule redact() documents."""

    def test_longer_match_wins_over_shorter_match_at_the_same_start(self) -> None:
        # SHORT ("aaa") finds two non-overlapping matches at [0,3) and [3,6);
        # LONG ("aaaaa") finds one match at [0,5) that starts at the same
        # position as SHORT's first hit but is longer. LONG must win the
        # position, which then knocks out *both* SHORT matches: the first
        # for sharing a start, the second for falling inside LONG's span.
        config = RedactionConfig(
            patterns=(
                RedactionPattern(name="SHORT", regex=r"aaa"),
                RedactionPattern(name="LONG", regex=r"aaaaa"),
            )
        )
        redactor = Redactor(config)

        result = redactor.redact("aaaaaa")

        assert result.text == "[LONG_1]a"
        assert dict(result.mapping) == {"[LONG_1]": "aaaaa"}

    def test_identical_span_from_two_patterns_resolves_by_declaration_order(self) -> None:
        # ALPHA and BETA match the exact same span with the exact same
        # length; only declaration order in config.patterns can break the
        # tie, and it must do so deterministically (stable sort).
        config = RedactionConfig(
            patterns=(
                RedactionPattern(name="ALPHA", regex=r"xyz"),
                RedactionPattern(name="BETA", regex=r"xyz"),
            )
        )
        redactor = Redactor(config)

        result = redactor.redact("xyz")

        assert result.text == "[ALPHA_1]"
        assert dict(result.mapping) == {"[ALPHA_1]": "xyz"}

    def test_declaration_order_reversed_flips_the_winner(self) -> None:
        # Same scenario as above with the two patterns declared in the
        # opposite order: the winner must flip too, proving the outcome is
        # driven by declaration order and not by name or regex content.
        config = RedactionConfig(
            patterns=(
                RedactionPattern(name="BETA", regex=r"xyz"),
                RedactionPattern(name="ALPHA", regex=r"xyz"),
            )
        )
        redactor = Redactor(config)

        result = redactor.redact("xyz")

        assert result.text == "[BETA_1]"


class TestTokenStabilityAndCounters:
    def test_two_distinct_values_in_one_category_get_incrementing_counters(self) -> None:
        redactor = Redactor(RedactionConfig())

        result = redactor.redact("alice@example.com and bob@example.org")

        assert result.text == "[EMAIL_1] and [EMAIL_2]"
        assert dict(result.mapping) == {
            "[EMAIL_1]": "alice@example.com",
            "[EMAIL_2]": "bob@example.org",
        }

    def test_same_value_repeated_in_one_call_reuses_its_token(self) -> None:
        redactor = Redactor(RedactionConfig())

        result = redactor.redact("alice@example.com wrote to alice@example.com again")

        assert result.text == "[EMAIL_1] wrote to [EMAIL_1] again"
        assert dict(result.mapping) == {"[EMAIL_1]": "alice@example.com"}

    def test_same_value_across_two_calls_reuses_its_token(self) -> None:
        redactor = Redactor(RedactionConfig())

        first = redactor.redact("alice@example.com")
        second = redactor.redact("bob@example.org and alice@example.com")

        assert first.text == "[EMAIL_1]"
        assert second.text == "[EMAIL_2] and [EMAIL_1]"
        assert dict(second.mapping) == {
            "[EMAIL_2]": "bob@example.org",
            "[EMAIL_1]": "alice@example.com",
        }

    def test_counters_are_independent_per_category(self) -> None:
        redactor = Redactor(RedactionConfig())

        result = redactor.redact("alice@example.com from 192.168.1.10 and 10.0.0.1")

        assert result.text == "[EMAIL_1] from [IPV4_1] and [IPV4_2]"


class TestRestore:
    def test_round_trip_on_mixed_default_pattern_content(self) -> None:
        text = (
            "Email alice@example.com, call +14155552671 or (415) 555-2671, "
            "server at 192.168.1.10, token ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123=="
        )
        redactor = Redactor(RedactionConfig())

        result = redactor.redact(text)

        assert "[EMAIL_1]" in result.text
        assert "[PHONE_E164_1]" in result.text
        assert "[PHONE_US_1]" in result.text
        assert "[IPV4_1]" in result.text
        assert "[SECRET_TOKEN_1]" in result.text
        mapping = dict(result.mapping)
        assert mapping["[EMAIL_1]"] == "alice@example.com"
        assert mapping["[PHONE_E164_1]"] == "+14155552671"
        assert mapping["[PHONE_US_1]"] == "(415) 555-2671"
        assert mapping["[IPV4_1]"] == "192.168.1.10"
        assert mapping["[SECRET_TOKEN_1]"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123=="  # noqa: S105 — redaction token fixture, not a credential

        assert redactor.restore(result.text) == text

    def test_restore_with_no_tokens_present_is_a_no_op(self) -> None:
        redactor = Redactor(RedactionConfig())
        text = "plain text, nothing to restore"
        assert redactor.restore(text) == text

    def test_unknown_counter_in_a_known_category_raises(self) -> None:
        redactor = Redactor(RedactionConfig())
        # EMAIL is a category this Redactor knows, but it never issued
        # EMAIL_1 (redact() was never called), so this must fail closed.
        with pytest.raises(UnknownRedactionTokenError, match="EMAIL"):
            redactor.restore("Hello [EMAIL_1], welcome")

    def test_bracket_text_in_an_unrecognized_category_is_left_untouched(self) -> None:
        redactor = Redactor(RedactionConfig())
        text = "See [NOTES_1] for the full context"
        assert redactor.restore(text) == text

    def test_token_from_one_instance_is_not_restorable_by_a_fresh_instance(self) -> None:
        source = Redactor(RedactionConfig())
        result = source.redact("alice@example.com")

        other = Redactor(RedactionConfig())
        with pytest.raises(UnknownRedactionTokenError):
            other.restore(result.text)

    def test_restore_is_pure_with_respect_to_prior_calls(self) -> None:
        # restore() must not itself allocate new tokens or mutate counters:
        # calling it does not make a previously-unknown token become known.
        redactor = Redactor(RedactionConfig())
        with pytest.raises(UnknownRedactionTokenError):
            redactor.restore("[EMAIL_1]")
        with pytest.raises(UnknownRedactionTokenError):
            redactor.restore("[EMAIL_1]")


class TestTokenFormatCollision:
    """Input text that already contains a literal, token-shaped substring.

    Documented, deliberate behavior (see Redactor.redact's docstring):
    redact() does not scan the input for pre-existing token-shaped text
    before allocating new tokens, so a literal collision is possible and,
    once it occurs, restore() cannot tell the two apart. This test pins
    that behavior so it is a known, observed characteristic rather than an
    untested surprise.
    """

    def test_preexisting_literal_token_collides_with_a_real_one(self) -> None:
        redactor = Redactor(RedactionConfig())
        # The first (and only) real email match becomes "[EMAIL_1]", which
        # is byte-identical to the literal bracket text already present.
        text = "Placeholder [EMAIL_1] next to a real alice@example.com address"

        result = redactor.redact(text)

        # Both the literal placeholder and the real match now read
        # identically in the output.
        assert result.text.count("[EMAIL_1]") == 2
        assert dict(result.mapping) == {"[EMAIL_1]": "alice@example.com"}

        # restore() cannot distinguish them: both occurrences are replaced,
        # so the previously-inert placeholder is corrupted into the email
        # address instead of being restored verbatim. This is the
        # documented cost of a human-readable bracket token format.
        restored = redactor.restore(result.text)
        assert restored == "Placeholder alice@example.com next to a real alice@example.com address"
        assert restored != text


class TestEdgeCases:
    def test_empty_text_produces_empty_result(self) -> None:
        redactor = Redactor(RedactionConfig())
        result = redactor.redact("")
        assert result.text == ""
        assert result.mapping == ()
        assert redactor.restore("") == ""

    def test_text_with_no_matches_is_unchanged(self) -> None:
        redactor = Redactor(RedactionConfig())
        text = "no sensitive data appears in this plain english sentence."
        result = redactor.redact(text)
        assert result.text == text
        assert result.mapping == ()

    def test_unicode_text_is_preserved_around_a_match(self) -> None:
        redactor = Redactor(RedactionConfig())
        text = "héllo wörld 😀 café bob@example.com résumé naïve 北京 東京"

        result = redactor.redact(text)

        assert "bob@example.com" not in result.text
        assert "café" in result.text
        assert "北京 東京" in result.text
        assert redactor.restore(result.text) == text
