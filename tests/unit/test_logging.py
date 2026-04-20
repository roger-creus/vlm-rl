from pathlib import Path


def test_csv_writer_creates_header_and_appends_rows(tmp_path: Path):
    from cleanrl_vlm.training.logging import CSV_COLUMNS, CsvWriter

    csv_path = tmp_path / "metrics.csv"
    w = CsvWriter(csv_path)
    w.log({"global_step": 1, "loss_total": 0.5, "gen_truncated_rate": 0.1})
    w.log({"global_step": 2, "lora_weight_norm_actor": 1.23, "inv_4_status": "green"})
    w.close()
    content = csv_path.read_text().splitlines()
    assert content[0].split(",") == CSV_COLUMNS
    assert len(content) == 3


def test_csv_schema_has_all_reviewer_m4_m5_fields():
    from cleanrl_vlm.training.logging import CSV_COLUMNS

    for col in [
        "gen_truncated_rate",
        "lora_weight_norm_actor",
        "lora_weight_norm_critic",
        "adapter_sync_wall_s",
        "inv_4_status",
    ]:
        assert col in CSV_COLUMNS, f"missing §9 column {col}"
