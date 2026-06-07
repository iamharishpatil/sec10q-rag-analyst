from pathlib import Path

from sec_rag.pipeline import PipelineConfig, PipelineRunner


def test_pipeline_config_default_resolves_project_paths() -> None:
    config = PipelineConfig.default()

    assert config.raw_dir == Path.cwd() / "data/raw/sec-10-q"
    assert config.pages_dir == Path.cwd() / "data/processed/pages"
    assert config.chunks_dir == Path.cwd() / "data/processed/chunks"
    assert config.qdrant_index_dir == Path.cwd() / "data/indexes/qdrant"
    assert config.bm25_index_dir == Path.cwd() / "data/indexes/bm25"


class RecordingPipelineRunner(PipelineRunner):
    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        self.calls: list[str] = []

    def parse_pages(self) -> tuple[int, int]:
        self.calls.append("parse")
        return 10, 1

    def chunk_pages(self) -> tuple[int, int]:
        self.calls.append("chunk")
        return 20, 1

    def build_indexes(self) -> tuple[int, int]:
        self.calls.append("index")
        return 20, 20

    def retrieve(self):
        self.calls.append("retrieve")
        return tuple()

    def evaluate(self) -> tuple[int, int, float]:
        self.calls.append("evaluate")
        return 5, 4, 0.8


def test_pipeline_runner_calls_enabled_stages_in_order(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config = PipelineConfig.default()
    config = PipelineConfig(
        raw_dir=raw_dir,
        pages_dir=config.pages_dir,
        chunks_dir=config.chunks_dir,
        qdrant_index_dir=config.qdrant_index_dir,
        bm25_index_dir=config.bm25_index_dir,
        qna_path=config.qna_path,
    )
    runner = RecordingPipelineRunner(config)

    result = runner.run()

    assert runner.calls == ["parse", "chunk", "index", "retrieve", "evaluate"]
    assert result.parsed_pages == 10
    assert result.chunks == 20
    assert result.recall_at_k == 0.8
