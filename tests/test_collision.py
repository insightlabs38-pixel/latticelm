from latticelm.collision import analyze


def test_collision_analysis_assigns_exact_and_hashed_regions() -> None:
    result = analyze([1, 2, 1, 2, 1, 2, 3, 4], exact_budgets=(2, 2, 2), tail_slots=8)
    assert result["orders"]["2"]["exact_entries"] == 2
    assert 0 <= result["orders"]["3"]["exact_hit_rate"] <= 1
