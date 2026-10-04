from collections import Counter
from datetime import date

import pytest

from src.masking import (
    HFEntityMasker,
    NameDictionary,
    age_band,
    fold,
    independent_residual_counts,
    load_ibge_first_names,
    mask_name_chains,
    mask_own_name,
    mask_patterns,
    mask_texts,
    name_tokens,
    residual_pattern_counts,
    surrogate_id,
    to_month,
)

# --- patterns -----------------------------------------------------------------------------


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
        ("call 98765 4321", "[PHONE]"),  # spaced mobile
        ("call 3333 4444", "[PHONE]"),  # spaced landline
        ("cpf 123.456.789-09", "[REDACTED]"),  # a labeled value is redacted whatever it is
        ("o documento 123.456.789-09", "[ID]"),
        ("cnpj 12.345.678/0001-95", "[ID]"),
        ("o rg 12 345 678 9 foi", "[REDACTED]"),
        ("documento 12 345 678 9 aqui", "[ID]"),  # spaced RG
        ("cep 01310-100", "[POSTCODE]"),
        ("born 05/03/1990", "[DATE]"),
        ("born 05-03-90", "[DATE]"),
        ("born 1990-04-02", "[DATE]"),  # ISO date
        ("id 12345678901", "[NUMBER]"),
        ("sou casado há anos", "[REDACTED]"),  # marital status, with or without a label
        ("tenho 2 filhos", "[REDACTED]"),
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
        "equipe separada por área",
    ],
)
def test_ordinary_text_with_numbers_is_left_alone(text):
    assert mask_patterns(text) == text


def test_pattern_stats_are_counted():
    stats: Counter = Counter()

    mask_patterns("a@b.com c@d.com (11) 98765-4321", stats)

    assert stats["EMAIL"] == 2
    assert stats["PHONE"] == 1


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
        ("Endereço = Rua das Flores 12", "Endereço", "Flores"),  # '=' separator
        ("Nome\nFulano de Tal\nExperiência", "Experiência", "Fulano"),  # no separator at all
        ("idade 32 e python", "python", "32"),  # label then number, no separator
        ("nascimento 12 03 1990 fim", "fim", "1990"),
        ("Estado civil: Casado, 2 filhos", "Estado civil:", "filhos"),
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
        "nome do projeto era alfa",  # a label word without separator inside a sentence
    ],
)
def test_labels_inside_other_words_do_not_trigger(text):
    assert mask_patterns(text) == text


def test_written_out_dates_and_ages_are_masked():
    assert mask_patterns("nascido em 5 de março de 1990") == "nascido em [DATE]"
    assert mask_patterns("tenho 32 anos de idade") == "tenho [AGE]"


def test_numeric_dates_glued_to_letters_are_masked():
    assert mask_patterns("nasc12/05/1990") == "nasc[DATE]"
    assert mask_patterns("em 12/05/1990a") == "em [DATE]a"


def test_independent_estimate_ignores_year_pairs_split_by_a_newline():
    counts = independent_residual_counts(["formacao\n2010\n2012", "tel 3333 4444"])

    assert counts["phone_like"] == 1  # only the real number


@pytest.mark.parametrize(
    "text",
    [
        "cel.: (11) 94547 - 9889",
        "(11) 94547– 9889",
        "tel (11)94547 – 9889",
        "11 94547 - 9889",
        "11) 9 4547 9889",
    ],
)
def test_phones_with_spaced_or_en_dash_separators_are_masked(text):
    assert "9889" not in mask_patterns(text)


def test_year_ranges_with_spaced_dashes_are_not_phones():
    for text in ("2015 - 2018", "2015 – 2018", "jan 2015 - 2018"):
        assert mask_patterns(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nascida em [DATE], 34 anos.", "nascida em [DATE], [AGE]."),
        ("brasileiro, 34 anos, [REDACTED]", "brasileiro, [AGE], [REDACTED]"),
        ("tenho 41 anos e moro em sp", "tenho [AGE] e moro em sp"),
        ("mais de 10 anos de experiência", "mais de 10 anos de experiência"),
        ("atuo há 20 anos na área", "atuo há 20 anos na área"),
        ("com 15 anos de mercado", "com 15 anos de mercado"),
        ("5 anos de experiência", "5 anos de experiência"),
    ],
)
def test_bare_ages_are_masked_only_in_a_personal_data_context(text, expected):
    assert mask_patterns(text) == expected


def test_children_and_family_phrases_are_masked():
    assert mask_patterns("casada, sem filhos") == "[REDACTED], [REDACTED]"
    assert mask_patterns("pai de dois filhos") == "pai de [REDACTED]"


def test_street_addresses_and_neighborhoods_are_masked_but_cities_and_parks_stay():
    assert mask_patterns("rua das flores, 123, campinas") == "[ADDRESS], campinas"
    assert mask_patterns("avenida paulista 1000 - sao paulo") == "[ADDRESS] - sao paulo"
    assert mask_patterns("mora no jardim são sebastião, hortolândia") == (
        "mora no [LOCATION], hortolândia"
    )
    assert mask_patterns("avenida [NAME], 1000, sp") == "[ADDRESS], sp"
    assert mask_patterns("parque tecnológico de são josé") == "parque tecnológico de são josé"


def test_redaction_is_idempotent_and_counted_once():
    stats: Counter = Counter()

    once = mask_patterns("Estado civil: Casado", stats)
    twice = mask_patterns(once, stats)

    assert once == twice == "Estado civil: [REDACTED]"
    assert stats["LABELED"] == 1


def test_self_consistency_check_reports_unredacted_labeled_values():
    assert residual_pattern_counts(["Estado civil: Casado"])["LABELED"] == 1
    assert residual_pattern_counts(["Estado civil: [REDACTED]"])["LABELED"] == 0


# --- independent residual check -----------------------------------------------------------


def test_independent_check_is_quiet_on_clean_text_and_year_ranges():
    clean = ["experiência 2015-2018 em python", "contato com a equipe de vendas", "tel aviv"]

    assert sum(independent_residual_counts(clean).values()) == 0


@pytest.mark.parametrize(
    ("leak", "check"),
    [
        ("fale com a@b.com", "email_or_at_sign"),
        ("nasceu em 1990-04-02", "iso_date"),
        ("ligue 98765 4321", "phone_like"),
        ("estado civil: casado", "labeled_value"),
        ("idade 32", "label_then_digits"),
        ("nome\nfulano de tal", "label_alone_then_text"),
        ("tem 2 filhos", "marital_or_children"),
    ],
)
def test_independent_check_catches_what_slipped_through(leak, check):
    assert independent_residual_counts([leak])[check] == 1


def test_independent_check_does_not_just_echo_the_masker():
    # An ISO date is exactly the kind of thing the (old) masker patterns missed; masking it
    # now means the independent check must see nothing left.
    assert independent_residual_counts([mask_patterns("nasceu em 1990-04-02")])["iso_date"] == 0


# --- accent folding and the person's own name ---------------------------------------------


def test_fold_keeps_the_length_so_offsets_stay_valid():
    text = "João Conceição ÁÉÍÓÚ çãõ İstanbul"

    assert len(fold(text)) == len(text)
    assert fold("João") == "joao"


def test_own_name_is_removed_case_and_accent_insensitively():
    masked = mask_own_name(
        "JOAO da Silva e joão trabalharam em casa da vovó. Silvana ficou.", "João da Silva"
    )

    assert masked.startswith("[NAME] e [NAME] trabalharam")  # the whole name goes as one unit
    assert "Silvana" in masked  # a different word that merely contains the token
    assert "casa da vovó" in masked  # an unrelated "da" is not touched


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("Joao Silva", "falei com João Silva ontem"),  # accent in the text, none in the record
        ("João Silva", "falei com joao silva ontem"),  # and the other way around
        ("JOÃO SILVA", "falei com joão silva ontem"),
    ],
)
def test_own_name_matching_is_accent_insensitive_in_both_directions(name, text):
    masked = mask_own_name(text, name)

    assert "silva" not in masked.lower() and "joão" not in masked.lower()
    assert "falei com" in masked


def test_short_names_are_masked_as_a_whole_phrase():
    assert mask_own_name("contato li wu hoje", "Li Wu") == "contato [NAME] hoje"


def test_own_name_stats_count_each_masked_span():
    stats: Counter = Counter()

    mask_own_name("maria souza e maria", "Maria Souza", stats)

    assert stats["OWN_NAME"] >= 2


def test_name_tokens_skip_particles_and_short_words():
    assert name_tokens("Maria de Fátima dos Santos Jr") == ["maria", "fatima", "santos"]
    assert name_tokens(None) == []


# --- name chains from a dictionary --------------------------------------------------------

D = NameDictionary.build(["Maria", "João", "Ana", "Paulo"], extra_tokens=["Silva", "Santos"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("falei com maria silva ontem", "falei com [NAME] ontem"),
        ("falei com joão da silva ontem", "falei com [NAME] ontem"),  # particle inside
        ("falei com joao silva santos hoje", "falei com [NAME] hoje"),  # accent-insensitive
        ("maria silva santos ana joao", "[NAME] joao"),  # a chain is capped at 4 tokens
        ("paulo silva", "[NAME]"),
    ],
)
def test_name_chains_are_masked(raw, expected):
    assert mask_name_chains(raw, D) == expected


@pytest.mark.parametrize(
    "text",
    [
        "falei com maria ontem",  # a lone first name is not a chain
        "maria\nsilva",  # a line break ends a chain
        "maria, silva",  # so does a comma: two different people
        "moro em são paulo silva",  # "são paulo" is a place, not a person
        "silva maria",  # a chain must START with a first name
        "analista de sistemas",
    ],
)
def test_name_chains_leave_non_names_alone(text):
    assert mask_name_chains(text, D) == text


def test_name_chains_do_not_mask_saint_places():
    places = NameDictionary.build(["santo", "santa", "andre", "catarina", "maria"], [])

    for text in ("moro em santo andré", "sede em santa catarina", "unidade santo andre sp"):
        assert mask_name_chains(text, places) == text
    assert mask_name_chains("falei com maria catarina", places) == "falei com [NAME]"


def test_unknown_surnames_after_a_first_name_are_masked_near_the_top_only():
    words = ["analista", "vendas", "gerente", "projetos"]
    open_dict = NameDictionary.build(["marcela", "maria"], ["silva"], known_words=words)

    assert mask_name_chains("marcela lucindo\nanalista de vendas", open_dict) == (
        "[NAME]\nanalista de vendas"
    )
    assert mask_name_chains("maria silva lucindo mesquita santos", open_dict) == "[NAME] santos"
    # a known word is never a surname, and a first name alone is not a chain
    assert mask_name_chains("maria gerente de projetos", open_dict) == "maria gerente de projetos"
    assert mask_name_chains("maria julho", open_dict) == "maria julho"  # months are ordinary words
    # deep in the text an unknown word after a first name is more likely an employer or a school
    deep = "analista de vendas " * 40 + "carlos lucindo"
    assert mask_name_chains(deep, open_dict) == deep
    # and without a vocabulary the rule is off
    plain = NameDictionary.build(["marcela"], [])
    assert mask_name_chains("marcela lucindo", plain) == "marcela lucindo"


def test_name_chain_stats_are_counted():
    stats: Counter = Counter()

    mask_name_chains("maria silva e joão santos", D, stats)

    assert stats["NAME_CHAIN"] == 2


def test_ibge_loader_takes_the_most_common_names_and_skips_blank_rows(tmp_path):
    csv_path = tmp_path / "names.csv"
    csv_path.write_text(
        "Nome,ate1930,ate1940\nMARIA,100,200\nRARO,1,\n\nANA,50,60\n,5,5\n", encoding="utf-8"
    )

    assert load_ibge_first_names(csv_path, top_k=2) == ["MARIA", "ANA"]


def test_dictionary_ignores_tokens_shorter_than_three_letters():
    d = NameDictionary.build(["Al", "Maria"], extra_tokens=["de"])

    assert "al" not in d.first_names and "maria" in d.first_names and "de" not in d.name_tokens


# --- NER (injected pipeline) --------------------------------------------------------------


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


def test_ner_never_masks_protected_terms_or_one_and_two_letter_spans():
    pipe = _fake_pipe([("COBOL", "PER"), ("R", "PER"), ("Nataly", "PER")])
    ner = HFEntityMasker(pipe=pipe, protected={"cobol"})

    [out] = ner.mask_many(["COBOL e R com Nataly"])

    assert out == "COBOL e R com [NAME]"  # technologies stay, the person goes


def test_ner_spans_are_split_at_protected_words():
    def pipe(chunks, batch_size, stride):
        # the model returns one span covering the job word and the name
        return [[{"entity_group": "PER", "score": 0.9, "start": 0, "end": len(c)}] for c in chunks]

    ner = HFEntityMasker(pipe=pipe, protected={"analista", "junior"})

    out = ner.mask_many(["Analista Junior", "Analista Maria Souza Junior"])

    assert out == ["Analista Junior", "Analista [NAME] Junior"]


def test_ner_chunking_loses_no_text_and_maps_results_back_per_text():
    ner = HFEntityMasker(chunk_chars=20, pipe=_fake_pipe([("Ana", "PER")]))
    long_text = "linha um com Ana\n" * 6 + "x" * 50  # several chunks plus an over-long line
    texts = [long_text, "", "Ana"]

    out = ner.mask_many(texts)

    assert out[0].replace("[NAME]", "Ana") == long_text
    assert out[1] == ""
    assert out[2] == "[NAME]"


def test_ner_never_cuts_a_name_in_half_at_a_chunk_boundary():
    # Regression: a hard cut at 800 characters split "Maria" into "Ma" + "ria", so neither
    # chunk contained the name and it survived.
    ner = HFEntityMasker(chunk_chars=800, pipe=_fake_pipe([("Maria", "PER")]))
    text = "x " * 399 + "Maria"  # one long line, the name lands right at the boundary

    chunks = ner._chunks(text)
    [out] = ner.mask_many([text])

    assert "".join(chunks) == text
    assert all("Mar" not in c or "Maria" in c for c in chunks)
    assert "Maria" not in out


def test_ner_failure_is_not_swallowed():
    def broken(chunks, batch_size, stride):
        raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError):
        HFEntityMasker(pipe=broken).mask_many(["Maria"])


def test_mask_texts_applies_all_layers_in_order():
    ner = HFEntityMasker(pipe=_fake_pipe([("Recife", "LOC")]))

    [out] = mask_texts(["Maria Souza, maria@x.com, Recife"], ["Maria Souza"], ner)

    assert out == "[NAME], [EMAIL], [LOC]"


def test_mask_texts_uses_the_dictionary_layer_and_runs_ner_once_per_unique_text():
    calls = []

    class Counting:
        counts: Counter = Counter()

        def mask_many(self, texts):
            calls.append(list(texts))
            return list(texts)

    out = mask_texts(["ana silva", "ana silva", "python"], None, Counting(), None, D)

    assert out == ["[NAME]", "[NAME]", "python"]
    assert calls == [["[NAME]", "python"]]  # duplicates were collapsed before the model


@pytest.mark.slow
def test_real_ner_model_masks_a_portuguese_name():
    [out] = HFEntityMasker().mask_many(["Maria da Silva trabalhou na Petrobras em São Paulo."])

    assert "Maria" not in out


# --- identifiers and generalization -------------------------------------------------------


def test_surrogate_id_is_stable_salted_and_namespaced():
    salt = b"s" * 32

    assert surrogate_id("cand", "42", salt) == surrogate_id("cand", "42", salt)
    assert surrogate_id("cand", "42", salt) != surrogate_id("cand", "42", b"t" * 32)
    assert surrogate_id("cand", "42", salt) != surrogate_id("vac", "42", salt)
    assert surrogate_id("cand", "42", salt) != "42"
    assert surrogate_id("cand", "", salt) == ""
    assert surrogate_id("cand", None, salt) == ""


def test_the_value_zero_is_a_real_id_not_a_missing_one():
    assert surrogate_id("cand", "0", b"s" * 32) != ""


def test_dates_are_generalized_to_the_month():
    assert to_month("05-03-2021") == "2021-03"
    assert to_month("1990-04-02") == "1990-04"
    assert to_month("") is None
    assert to_month("not a date") is None


def test_age_band_uses_the_attained_age_on_the_snapshot_date():
    snapshot = date(2026, 10, 4)

    assert age_band("2001-01-01", snapshot) == "25-34"  # already turned 25
    assert age_band("2001-12-31", snapshot) == "<25"  # turns 25 later in the year
    assert age_band("1990-04-02", snapshot) == "35-44"
    assert age_band("1960-01-01", snapshot) == "55+"
    assert age_band("2015-01-01", snapshot) is None  # implausible -> missing, not kept
    assert age_band("", snapshot) is None


def test_age_band_moves_with_the_snapshot_not_a_hardcoded_year():
    assert age_band("2001-01-01", date(2025, 6, 1)) == "<25"
    assert age_band("2001-01-01", date(2026, 6, 1)) == "25-34"
