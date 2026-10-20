from __future__ import annotations

from tokenserve.tokenizer import ByteTokenizer, IncrementalDetokenizer, StopStringMatcher

TOK = ByteTokenizer()


def test_byte_tokenizer_roundtrip_and_eos() -> None:
    text = "héllo, 世界"
    ids = TOK.encode(text)
    assert len(ids) == len(text.encode())
    assert TOK.decode(ids) == text
    assert TOK.decode([*ids, 256]) == text  # EOS renders as nothing
    assert TOK.is_special(256)
    assert not TOK.is_special(255)


def test_chat_template() -> None:
    ids = TOK.encode_chat([{"role": "user", "content": "hi"}])
    assert TOK.decode(ids) == "<|user|>\nhi\n<|assistant|>\n"


def test_incremental_detokenizer_never_emits_partial_characters() -> None:
    text = "naïve café → 日本語 ok"
    detok = IncrementalDetokenizer(TOK, TOK.encode("prompt: "))
    deltas = [detok.add(t) for t in TOK.encode(text)]
    assert "".join(deltas) == text
    assert not any("�" in d for d in deltas)
    assert deltas.count("") > 0  # multi-byte chars were held until complete
    assert detok.add(256) == ""


def test_stop_matcher_holds_back_possible_prefix_and_truncates() -> None:
    m = StopStringMatcher(["END"])
    out = [m.push(c) for c in "abcEN"]
    assert "".join(t for t, _ in out) == "abc"  # "EN" held back
    assert all(not stopped for _, stopped in out)
    assert m.push("D") == ("", True)
    assert m.text == "abc"


def test_stop_matcher_releases_false_alarm_and_picks_earliest_stop() -> None:
    m = StopStringMatcher(["END", "xy"])
    assert m.push("aEN") == ("a", False)  # "EN" could still become "END"
    assert m.push("o") == ("ENo", False)  # it did not: release it
    assert m.push("zz") == ("zz", False)  # nothing that could start a stop: no delay
    assert m.push("xyzEND") == ("", True)  # "xy" occurs before "END"
    assert m.text == "aENozz"


def test_stop_matcher_without_stops_passes_through() -> None:
    m = StopStringMatcher([])
    assert m.push("abc") == ("abc", False)
    assert m.flush() == ""
