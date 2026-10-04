from collections import Counter

import pytest

from src.masking import (
    HFEntityMasker,
    age_band,
    mask_own_name,
    mask_patterns,
    mask_texts,
    name_tokens,
    residual_pattern_counts,
    surrogate_id,
    to_month,
)


@pytest.mark.parametrize(
    ("raw", "placeholder"),
    [
        ("write to maria.silva@example.com now", "[EMAIL]"),
        ("see https://www.linkedin.com/in/maria-silva", "[URL]"),
        ("see linkedin.com/in/maria-silva", "[URL]"),
        ("call (11) 98765-4321", "[PHONE]"),
        ("call +55 11 98765-4321", "[PHONE]"),
        ("call 11 98765-4321", "[PHONE]"),
        ("call 3333-4444", "[PHONE]"),
        ("cpf 123.456.789-09", "[ID]"),
        ("cnpj 12.345.678/0001-95", "[ID]"),
        ("cep 01310-100", "[POSTCODE]"),
        ("born 05/03/1990", "[DATE]"),
        ("born 05-03-90", "[DATE]"),
        ("id 12345678901", "[NUMBER]"),
    ],
)
def test_patterns_are_masked(raw, placeholder):
    masked = mask_patterns(raw)

    assert placeholder in masked
    assert not any(ch.isdigit() for ch in masked)  # nothing numeric survives


@pytest.mark.parametrize(
    "text",
    [
        "Analista 2015-2018",
        "Analista (2018-2019)",
        "Analista 2015 2018",
        "Desde 03/2019 até jan/2021",
        "Python 3.12 e SQL",
        "Gerenciou 120 pessoas e 15 projetos",
    ],
)
def test_ordinary_text_with_numbers_is_left_alone(text):
    assert mask_patterns(text) == text


def test_pattern_stats_are_counted():
    stats: Counter = Counter()

    mask_patterns("a@b.com c@d.com (11) 98765-4321", stats)

    assert stats["EMAIL"] == 2
    assert stats["PHONE"] == 1


def test_own_name_is_removed_case_and_accent_insensitively():
    masked = mask_own_name("JOAO da Silva e joão trabalharam. Silvana ficou.", "João da Silva")

    assert "JOAO" not in masked and "Silva " not in masked and "joão" not in masked
    assert "Silvana" in masked  # a different word that merely contains the token
    assert " da " in masked  # particles are not treated as name tokens


def test_name_tokens_skip_particles_and_short_words():
    assert name_tokens("Maria de Fátima dos Santos Jr") == ["Maria", "Fátima", "Santos"]
    assert name_tokens(None) == []


def test_residual_check_finds_what_the_masker_missed():
    clean = ["nothing here", "Analista 2015-2018"]
    leaky = ["mail me: x@y.com"]

    assert sum(residual_pattern_counts(clean).values()) == 0
    assert residual_pattern_counts(leaky)["EMAIL"] == 1


def test_surrogate_id_is_stable_salted_and_namespaced():
    salt = b"s" * 32

    assert surrogate_id("cand", "42", salt) == surrogate_id("cand", "42", salt)
    assert surrogate_id("cand", "42", salt) != surrogate_id("cand", "42", b"t" * 32)
    assert surrogate_id("cand", "42", salt) != surrogate_id("vac", "42", salt)
    assert surrogate_id("cand", "42", salt) != "42"
    assert surrogate_id("cand", "", salt) == ""


def test_dates_are_generalized():
    assert to_month("05-03-2021") == "2021-03"
    assert to_month("1990-04-02") == "1990-04"
    assert to_month("") is None
    assert to_month("not a date") is None
    assert age_band("1990-04-02") == "35-44"
    assert age_band("2015-01-01") is None  # implausible age -> missing, not kept
    assert age_band("") is None


def _fake_pipe(targets):
    """A stand-in NER pipeline that tags every occurrence of the given (word, group) pairs."""

    def pipe(chunks, batch_size, stride):
        out = []
        for chunk in chunks:
            found = []
            for word, group in targets:
                start = chunk.find(word)
                while start != -1:
                    found.append(
                        {
                            "entity_group": group,
                            "score": 0.99,
                            "start": start,
                            "end": start + len(word),
                        }
                    )
                    start = chunk.find(word, start + 1)
            out.append(found)
        return out

    return pipe


def test_ner_masks_requested_types_only_and_counts_them():
    pipe = _fake_pipe([("Maria", "PER"), ("Petrobras", "ORG"), ("Recife", "LOC")])
    ner = HFEntityMasker(entity_types=("PER", "LOC"), pipe=pipe)

    [out] = ner.mask_many(["Maria trabalhou na Petrobras em Recife."])

    assert out == "[NAME] trabalhou na Petrobras em [LOC]."  # the organization is kept
    assert ner.counts == {"PER": 1, "LOC": 1}


def test_ner_low_confidence_entities_follow_the_threshold():
    def pipe(chunks, batch_size, stride):
        return [[{"entity_group": "PER", "score": 0.2, "start": 0, "end": 5}] for _ in chunks]

    assert HFEntityMasker(pipe=pipe, min_score=0.35).mask_many(["Maria"]) == ["Maria"]
    assert HFEntityMasker(pipe=pipe, min_score=0.1).mask_many(["Maria"]) == ["[NAME]"]


def test_ner_chunking_loses_no_text_and_maps_results_back_per_text():
    ner = HFEntityMasker(chunk_chars=20, pipe=_fake_pipe([("Ana", "PER")]))
    long_text = "linha um com Ana\n" * 6 + "x" * 50  # several chunks plus an over-long line
    texts = [long_text, "", "Ana"]

    out = ner.mask_many(texts)

    assert out[0].replace("[NAME]", "Ana") == long_text
    assert out[1] == ""
    assert out[2] == "[NAME]"


def test_ner_failure_is_not_swallowed():
    def broken(chunks, batch_size, stride):
        raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError):
        HFEntityMasker(pipe=broken).mask_many(["Maria"])


def test_mask_texts_applies_all_layers_in_order():
    ner = HFEntityMasker(pipe=_fake_pipe([("Recife", "LOC")]))

    [out] = mask_texts(["Maria Souza, maria@x.com, Recife"], ["Maria Souza"], ner)

    assert out == "[NAME] [NAME], [EMAIL], [LOC]"


@pytest.mark.slow
def test_real_ner_model_masks_a_portuguese_name():
    [out] = HFEntityMasker().mask_many(["Maria da Silva trabalhou na Petrobras em São Paulo."])

    assert "Maria" not in out


@pytest.mark.parametrize(
    ("raw", "kept", "gone"),
    [
        ("Estado civil: Casado\nPython", "Estado civil:", "Casado"),
        ("Data de nascimento: 12 de março de 1990", "Data de nascimento:", "março"),
        ("Bairro: Vila Mariana, São Paulo", "Bairro:", "Vila"),
        ("Idade: 32 anos, Solteiro, 5 anos de experiência", "5 anos de experiência", "32"),
        ("Nome:\nFulano de Tal\nExperiência", "Experiência", "Fulano"),
        ("NOME - Fulano de Tal", "NOME", "Fulano"),
        ("Nacionalidade: Brasileira | Cidade: Recife", "Nacionalidade:", "Recife"),
    ],
)
def test_labeled_personal_data_values_are_redacted(raw, kept, gone):
    masked = mask_patterns(raw)

    assert kept in masked and gone not in masked
    assert "[REDACTED]" in masked


@pytest.mark.parametrize(
    "text",
    [
        "Experiência em Intel: drivers",  # 'tel' inside a word is not a label
        "País: Brasil",  # 'pai' inside a word is not a label
        "Qualidade: alta, 10 anos de experiência",  # 'idade' inside a word
        "Python: avançado\nSQL: intermediário",
    ],
)
def test_labels_inside_other_words_do_not_trigger(text):
    assert mask_patterns(text) == text


def test_written_out_dates_and_ages_are_masked():
    assert mask_patterns("nascido em 5 de março de 1990") == "nascido em [DATE]"
    assert mask_patterns("tenho 32 anos de idade") == "tenho [AGE]"


def test_redaction_is_idempotent_and_counted_once():
    stats: Counter = Counter()

    once = mask_patterns("Estado civil: Casado", stats)
    twice = mask_patterns(once, stats)

    assert once == twice == "Estado civil: [REDACTED]"
    assert stats["LABELED"] == 1


def test_residual_check_reports_unredacted_labeled_values():
    assert residual_pattern_counts(["Estado civil: Casado"])["LABELED"] == 1
    assert residual_pattern_counts(["Estado civil: [REDACTED]"])["LABELED"] == 0
