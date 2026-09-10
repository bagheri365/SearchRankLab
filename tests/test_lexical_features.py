import numpy as np

from searchranklab.routing import (
    LEXICAL_FEATURE_NAMES,
    PreRetrievalRouter,
    build_lexical_statistics,
    lexical_query_features,
)


def test_lexical_statistics_count_document_frequency_once_per_document():
    corpus = {
        "d1": {"title": "", "text": "alpha alpha beta"},
        "d2": {"title": "", "text": "beta gamma"},
    }

    stats = build_lexical_statistics(corpus)

    assert stats.document_count == 2
    assert stats.document_frequency["alpha"] == 1
    assert stats.document_frequency["beta"] == 2


def test_rare_terms_receive_higher_idf_features():
    corpus = {
        "d1": {"title": "", "text": "common rare"},
        "d2": {"title": "", "text": "common"},
        "d3": {"title": "", "text": "common"},
    }
    stats = build_lexical_statistics(corpus)

    common = lexical_query_features("common", stats)
    rare = lexical_query_features("rare", stats)

    assert rare[0] > common[0]


def test_lexical_query_features_have_fixed_shape():
    stats = build_lexical_statistics(
        {"d1": {"title": "", "text": "alpha beta"}}
    )

    values = lexical_query_features("alpha unknown", stats)

    assert values.shape == (len(LEXICAL_FEATURE_NAMES),)
    assert np.isfinite(values).all()


def test_router_can_use_lexical_statistics():
    corpus = {
        "d1": {"title": "", "text": "common rare"},
        "d2": {"title": "", "text": "common"},
        "d3": {"title": "", "text": "common"},
    }
    stats = build_lexical_statistics(corpus)

    router = PreRetrievalRouter(
        class_weight=None,
        lexical_stats=stats,
    )

    assert router.lexical_stats is stats
