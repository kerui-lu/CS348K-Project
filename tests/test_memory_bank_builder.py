from build_memory_bank import build_parser, run_memory_bank_build


class FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, output_text: str):
        self.responses = FakeResponses(output_text)


def test_memory_bank_builder_writes_raw_and_reflection_files(tmp_path):
    raw_path = tmp_path / "raw.json"
    heuristic_path = tmp_path / "heuristics.json"
    results_dir = tmp_path / "results"
    args = build_parser().parse_args(
        [
            "--episodes",
            "1",
            "--max_steps",
            "1",
            "--levels",
            "levels/v2_pilot.json",
            "--results_dir",
            str(results_dir),
            "--raw_memory_path",
            str(raw_path),
            "--heuristic_memory_path",
            str(heuristic_path),
            "--max_llm_calls",
            "5",
        ]
    )
    agent_client = FakeClient("teleport")
    reflection_client = FakeClient('["Check box reachability before pushing."]')

    summary = run_memory_bank_build(args, agent_client=agent_client, reflection_client=reflection_client)

    assert raw_path.exists()
    assert heuristic_path.exists()
    assert (results_dir / "summary.json").exists()
    assert summary["raw_memory_item_count"] == 1
    assert summary["heuristic_count"] == 1
    assert summary["source_train_level_ids"] == ["simple_001", "corner_trap_001"]
    assert len(agent_client.responses.calls) == 1
    assert len(reflection_client.responses.calls) == 1
