"""Tests for goal auto-progress (POST /api/intelligence/goals/{id}/refresh).

Uses a real on-disk SQLite dataset + real run_explore aggregation — no mocks.
Covers: refresh updates current from live data; missing metric fails honestly
with the stored goal unchanged; unknown metric fails honestly.
"""

import json
import sqlite3

import pytest


USER_ID = 999881


@pytest.fixture
def live_session(tmp_path):
    """Real SQLite DB with order revenue + matching schema mapping."""
    db_path = tmp_path / "goal_refresh.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, revenue REAL, "
        "product_category TEXT, order_date DATE)"
    )
    rows = [
        (1, 100.0, "Bikes", "2024-01-15"),
        (2, 200.0, "Bikes", "2024-02-15"),
        (3, 300.0, "Gear", "2024-03-15"),
        (4, 400.0, "Gear", "2024-04-15"),
    ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()

    mapping = {
        "tables": {
            "orders": {
                "columns": [
                    {"column": "order_id", "data_type": "integer", "nullable": False},
                    {"column": "revenue", "data_type": "numeric", "nullable": True},
                    {"column": "product_category", "data_type": "text", "nullable": True},
                    {"column": "order_date", "data_type": "date", "nullable": True},
                ],
                "primary_key": ["order_id"],
                "row_count": 4,
            }
        }
    }
    schema_path = tmp_path / "schema_mapping_latest.json"
    schema_path.write_text(json.dumps(mapping), encoding="utf-8")
    return {
        "provider": "local",
        "source_type": "semi-structured",
        "db_uri": f"sqlite:///{db_path}",
        "db_name": "goal_refresh_ds",
        "schema_path": str(schema_path),
    }


@pytest.fixture
def dataset_id(live_session):
    from services.intelligence_service import _dataset_id

    return _dataset_id(live_session)


@pytest.fixture
def cleanup_goals():
    created = []
    yield created
    from services import intelligence_service as intel

    for gid in created:
        try:
            intel.delete_goal(USER_ID, gid)
        except Exception:
            pass


class TestGoalRefresh:
    def test_refresh_updates_current_from_data(
        self, live_session, dataset_id, cleanup_goals
    ):
        from services import intelligence_service as intel

        goal = intel.create_goal(
            USER_ID,
            dataset_id,
            {"title": "Grow revenue", "metric": "revenue",
             "target_value": 2000, "current_value": 10},
        )
        cleanup_goals.append(goal["id"])
        assert goal["current_value"] == 10

        result = intel.refresh_goal(USER_ID, dataset_id, goal["id"], live_session)
        assert result["ok"] is True, result.get("detail")
        updated = result["item"]
        # Live total = 100+200+300+400 = 1000.
        assert updated["current_value"] == pytest.approx(1000.0)
        assert updated["current"] == pytest.approx(1000.0)
        # progress = current/target recomputed by the backend.
        assert updated["progress_pct"] == pytest.approx(50.0)
        assert updated["status_label"] in (
            "Behind target", "Near target", "On track", "Achieved")
        assert updated.get("last_refresh_total") == pytest.approx(1000.0)

        # Stored goal persisted the refresh.
        stored = [g for g in intel.list_goals(USER_ID, dataset_id, "all")
                  if g["id"] == goal["id"]]
        assert stored and stored[0]["current_value"] == pytest.approx(1000.0)

    def test_missing_metric_honest_fail(
        self, live_session, dataset_id, cleanup_goals
    ):
        from services import intelligence_service as intel

        goal = intel.create_goal(USER_ID, dataset_id, {"title": "No metric yet"})
        cleanup_goals.append(goal["id"])

        result = intel.refresh_goal(USER_ID, dataset_id, goal["id"], live_session)
        assert result["ok"] is False
        assert result.get("code") == "METRIC_MISSING"

        # Goal unchanged in storage.
        stored = [g for g in intel.list_goals(USER_ID, dataset_id, "all")
                  if g["id"] == goal["id"]]
        assert stored and stored[0].get("current_value") in (None, 0)

    def test_unknown_metric_honest_fail(
        self, live_session, dataset_id, cleanup_goals
    ):
        from services import intelligence_service as intel

        goal = intel.create_goal(
            USER_ID, dataset_id,
            {"title": "Bad metric", "metric": "no_such_column",
             "target_value": 100, "current_value": 5},
        )
        cleanup_goals.append(goal["id"])

        result = intel.refresh_goal(USER_ID, dataset_id, goal["id"], live_session)
        assert result["ok"] is False
        stored = [g for g in intel.list_goals(USER_ID, dataset_id, "all")
                  if g["id"] == goal["id"]]
        assert stored and stored[0]["current_value"] == pytest.approx(5.0)

    def test_goal_not_found(self, live_session, dataset_id):
        from services import intelligence_service as intel

        result = intel.refresh_goal(
            USER_ID, dataset_id, "does-not-exist", live_session)
        assert result["ok"] is False
        assert result.get("code") == "NOT_FOUND"


class TestGoalStatusThresholds:
    """Refresh-time status pill thresholds shared with the frontend."""

    def _make(self, intel, dataset_id, current, target=100.0):
        return intel.create_goal(
            USER_ID, dataset_id,
            {"title": f"g-{current}", "metric": "revenue",
             "target_value": target, "current_value": current},
        )

    def test_threshold_labels(self, dataset_id, cleanup_goals):
        from services import intelligence_service as intel

        cases = [
            (100.0, "Achieved"),
            (90.0, "Near target"),
            (60.0, "On track"),
            (10.0, "Behind target"),
        ]
        for current, label in cases:
            goal = self._make(intel, dataset_id, current)
            cleanup_goals.append(goal["id"])
            assert goal["status_label"] == label, (current, goal)
