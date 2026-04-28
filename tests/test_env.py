from sokoban_memory.env import SokobanEnv
from sokoban_memory.types import Level, Position


def make_level(grid: list[str], level_id: str = "test") -> Level:
    walls = set()
    targets = set()
    boxes = set()
    player = None
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            pos = Position(r, c)
            if char == "#":
                walls.add(pos)
            elif char == ".":
                targets.add(pos)
            elif char == "$":
                boxes.add(pos)
            elif char == "*":
                boxes.add(pos)
                targets.add(pos)
            elif char == "@":
                player = pos
            elif char == "+":
                player = pos
                targets.add(pos)
    assert player is not None
    return Level(level_id, len(grid[0]), len(grid), walls, targets, boxes, player)


def test_player_moves_to_empty_cell():
    env = SokobanEnv(make_level([
        "#####",
        "#@  #",
        "# $.#",
        "#####",
    ]))
    env.step("Right")
    assert env.player == Position(1, 2)


def test_push_box():
    env = SokobanEnv(make_level([
        "#####",
        "#@$.#",
        "#   #",
        "#####",
    ]))
    result = env.step("Right")
    assert result.info["pushed_box"] is True
    assert env.player == Position(1, 2)
    assert Position(1, 3) in env.boxes
    assert env.is_solved()


def test_cannot_pull_box():
    env = SokobanEnv(make_level([
        "#####",
        "#$@.#",
        "#   #",
        "#####",
    ]))
    env.step("Right")
    assert Position(1, 1) in env.boxes
    assert env.player == Position(1, 3)


def test_cannot_push_two_boxes():
    env = SokobanEnv(make_level([
        "######",
        "#@$$.#",
        "#    #",
        "######",
    ]))
    result = env.step("Right")
    assert result.info["moved"] is False
    assert result.info["hit"] == "box_blocked_by_box"
    assert env.player == Position(1, 1)


def test_cannot_push_box_into_wall():
    env = SokobanEnv(make_level([
        "#####",
        "# @$#",
        "#  .#",
        "#####",
    ]))
    result = env.step("Right")
    assert result.info["moved"] is False
    assert result.info["hit"] == "box_blocked_by_wall_or_boundary"


def test_cannot_walk_into_wall():
    env = SokobanEnv(make_level([
        "#####",
        "#@$.#",
        "#   #",
        "#####",
    ]))
    result = env.step("Left")
    assert result.info["moved"] is False
    assert result.info["hit"] == "wall_or_boundary"


def test_solved_detection_for_box_on_target():
    env = SokobanEnv(make_level([
        "#####",
        "#@* #",
        "#   #",
        "#####",
    ]))
    assert env.is_solved()


def test_simple_deadlock_for_non_target_corner():
    env = SokobanEnv(make_level([
        "#####",
        "#$@.#",
        "#   #",
        "#####",
    ]))
    deadlocked, reason = env.is_deadlocked()
    assert deadlocked
    assert reason == "box_at_non_target_corner:1,1"


def test_box_on_target_corner_is_not_deadlock():
    env = SokobanEnv(make_level([
        "#####",
        "#*@ #",
        "#   #",
        "#####",
    ]))
    assert env.is_deadlocked() == (False, None)

