import numpy as np

from silent_dissent import metrics as M

L = list("ABCD")


def rec(original, stated, rows, **kw):
    return {"letters": L, "original": original, "stated": stated, "lens_logits": rows, **kw}


def test_silent_dissent_definitions():
    # 3 layers; original A, stated C.
    rows = [[3.0, 0, 1.0, 0], [2.0, 0, 1.5, 0], [0.0, 0, 5.0, 0]]
    r = rec("A", "C", rows)
    assert M.is_silent_dissent(r, 0, "strict")
    assert M.is_silent_dissent(r, 1, "strict")
    assert not M.is_silent_dissent(r, 2, "strict")
    assert not M.is_silent_dissent(r, 2, "lenient", delta=1.0)
    assert M.is_silent_dissent(r, 2, "lenient", delta=6.0)
    assert not M.is_silent_dissent(rec("A", "A", rows), 0)  # no flip, no dissent


def test_decision_flip_layer():
    rows = [[0, 0, 1, 0], [2, 0, 1, 0], [0, 0, 1, 0], [0, 0, 2, 0]]
    assert M.decision_flip_layer(rec("A", "C", rows)) == 2
    assert M.decision_flip_layer(rec("A", "A", rows)) is None


def test_select_layer():
    base = [rec("B", "B", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]) for _ in range(10)]
    assert M.select_layer(base, 0.9) == 1
    np.testing.assert_allclose(M.agreement_curve(base), [0, 1, 1])


def test_aggregation_internal_vote_recovers_dissent():
    # Gold A. Round 0: agents say A, A, B. Round 1: all say B but internally prefer A.
    lens_a = [[5, 0, 0, 0]]
    recs = []
    for a, s0 in enumerate("AAB"):
        recs.append({"item_id": "x", "round": 0, "gold": "A", "letters": L, "stated": s0,
                     "final_logits": [1, 0, 0, 0], "lens_logits": lens_a})
        recs.append({"item_id": "x", "round": 1, "gold": "A", "letters": L, "stated": "B",
                     "final_logits": [0, 1, 0, 0], "lens_logits": lens_a})
    t = M.aggregation_table(recs, layer=0).set_index("round")
    assert t.loc[1, "acc_stated"] == 0 and t.loc[1, "acc_internal"] == 1 and t.loc[1, "acc_initial"] == 1
