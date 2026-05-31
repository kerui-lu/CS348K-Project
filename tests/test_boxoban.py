from sokoban_memory.boxoban import canonical_grid_hash, is_standard_boxoban_grid, parse_boxoban_text
from sokoban_memory.solver import estimate_solution_cost, legal_pushes
from sokoban_memory.levels import load_levels


def test_parse_boxoban_text_preserves_spaces_and_indices():
    text = "\n".join(
        [
            "; 0",
            "##########",
            "#     #  #",
            "# $   .  #",
            "# @      #",
            "#        #",
            "# $   .  #",
            "#        #",
            "# $ $.   #",
            "#     .  #",
            "##########",
            "; 7",
            "##########",
            "#@ $.    #",
            "#  $ .   #",
            "#        #",
            "#  $ .   #",
            "#        #",
            "#  $ .   #",
            "#        #",
            "#        #",
            "##########",
        ]
    )

    puzzles = parse_boxoban_text(text)

    assert [puzzle.source_index for puzzle in puzzles] == [0, 7]
    assert puzzles[0].grid[1] == "#     #  #"
    assert is_standard_boxoban_grid(puzzles[0].grid)


def test_solver_estimates_simple_level_cost():
    level = next(level for level in load_levels("levels/v2_pilot.json") if level.level_id == "simple_001")

    result = estimate_solution_cost(level, max_states=1000)

    assert result.status == "solved"
    assert result.min_pushes == 1
    assert result.min_steps is not None
    assert legal_pushes(level)


def test_canonical_grid_hash_is_stable():
    grid = ["#####", "#@$.#", "#####"]

    assert canonical_grid_hash(grid) == canonical_grid_hash(list(grid))
