from sokoban_memory.levels import load_levels
from sokoban_memory.solver import bfs_solve, verify_solution


def test_bfs_solves_simple_calibration_levels():
    levels = load_levels("levels/v2_pilot.json")
    for level_id in ("simple_001", "corner_trap_001", "wall_push_001"):
        level = next(level for level in levels if level.level_id == level_id)
        result = bfs_solve(level, max_nodes=50_000)
        assert result.solution is not None, (level_id, result.reason)
        assert verify_solution(level, result.solution)
        assert level.reference_solution is not None
        assert len(result.solution) == level.optimal_steps


def test_secondary_references_file_has_solved_medium_levels():
    import json
    from pathlib import Path

    path = Path("levels/v2_pilot_secondary_references.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    solutions = data["solutions"]
    assert len(solutions) == 12
    for level_id, entry in solutions.items():
        assert entry["status"] == "solved", level_id
        assert entry["verified_in_env"] is True
        assert len(entry["reference_solution"]) == entry["steps"]
