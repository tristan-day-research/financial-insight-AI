import pytest
from pathlib import Path
from knowledge_base.src.ingestion.sec_sql_extractor import SECDataExtractor


class TestSECDataExtractor:
    def test_validate_xbrl_file(self, tmp_path):
        extractor = SECDataExtractor()
        xbrl_content = """<?xml version='1.0' encoding='UTF-8'?>
<xbrl xmlns='http://www.xbrl.org/2003/instance'></xbrl>
"""
        xbrl_file = tmp_path / "sample.xbrl"
        xbrl_file.write_text(xbrl_content)
        assert extractor._validate_xbrl_file(xbrl_file) is True
