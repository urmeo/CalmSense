"""Project logging preserves host configuration and supports destination changes."""

import json
import logging

import pytest
import structlog

from src import logging_config


def test_logging_reconfigures_destinations_without_resetting_host(tmp_path, monkeypatch):
    project = logging.getLogger("calmsense")
    saved = (project.handlers[:], project.level, project.propagate)
    project.handlers = []
    host = logging.getLogger()
    host_handler = logging.NullHandler()
    host.addHandler(host_handler)
    structlog_before = structlog.get_config().copy()
    monkeypatch.setattr(logging_config, "_logging_configured", False)
    monkeypatch.setattr(logging_config, "_configured_settings", None)
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    try:
        logging_config.setup_logging(log_file=first, console=False)
        logger = logging_config.get_logger("src.test")
        logger.info("first", value=1)
        old_handler = project.handlers[0]
        logging_config.setup_logging(log_file=second, console=False)
        logger.info("second", value=2)
        assert old_handler.stream is None
        assert json.loads(first.read_text())["event"] == "first"
        assert json.loads(second.read_text())["event"] == "second"
        assert host_handler in host.handlers
        assert structlog.get_config() == structlog_before
    finally:
        for handler in project.handlers:
            handler.close()
        project.handlers, project.level, project.propagate = saved
        host.removeHandler(host_handler)


def test_invalid_logging_level_is_rejected():
    with pytest.raises(ValueError, match="Unknown logging level"):
        logging_config.setup_logging(level="invalid", log_file=None)
