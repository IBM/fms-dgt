# Third Party
from dotenv import load_dotenv

# Local
from fms_dgt.core.datastores.default import DefaultDatastore

load_dotenv()


class TestDefault:
    def test_text_txt(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/sample.txt",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 1

        assert "This is a sample file in text format." == items[0]

    def test_text_md(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/sample.md",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 1

        assert "This is a sample file in markdown format." == items[0]

    def test_single_item_json(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/single_item.json",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 1

        assert {"key_1": "value_1", "key_2": {"key_2.key_a": "value_2a"}} == items[0]

    def test_multiple_items_json(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/multiple_items.json",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 2

        assert {"key_1": "value_1", "key_2": {"key_2.key_a": "value_2a"}} == items[0]
        assert {"key_a": "value_a", "key_b": {"key_b.key_1": "value_b1"}} == items[1]

    def test_single_jsonl(self):
        primary_cfg = {
            "store_name": "test",
            "data_path": "tests/data/core/datastores/single_item_1.jsonl",
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 1

        assert "content" in items[0]
        assert {"id": 1, "content": "This is sample content 1"} == items[0]

    def test_multiple_jsonl(self):
        primary_cfg = {
            "store_name": "test",
            "data_path": "tests/data/core/datastores/single_item_*.jsonl",
            "data_formats": ["jsonl"],
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 2

        items = datastore.load_data()
        assert len(items) == 2

        assert "content" in items[0]

    def test_single_item_yaml(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/single_item.yaml",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 1

        assert {"key_1": "value_1", "key_2": {"key_2.key_a": "value_2a"}} == items[0]

    def test_multiple_items_yaml(self):
        datastore = DefaultDatastore(
            type="default",
            **{
                "store_name": "test",
                "data_path": "tests/data/core/datastores/multiple_items.yaml",
            },
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 2

        assert {"key_1": "value_1", "key_2": {"key_2.key_a": "value_2a"}} == items[0]
        assert {"key_a": "value_a", "key_b": {"key_b.key_1": "value_b1"}} == items[1]

    def test_parquet(self):
        primary_cfg = {
            "store_name": "knowledge",
            "data_path": "tests/data/core/datastores/multiple_items_*",
            "data_formats": ["parquet"],
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 2

        items = datastore.load_data()
        assert len(items) == 10

    def test_csv(self):
        primary_cfg = {
            "store_name": "test",
            "data_path": "tests/data/core/datastores/multiple_item_1.csv",
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 3

        assert ["1", ' "This is sample content 1"'] == items[0]
        assert ["2", ' "This is sample content 2"'] == items[1]
        assert ["3", ' "This is sample content 3"'] == items[2]

    def test_csv_with_optional_arguments(self):
        primary_cfg = {
            "store_name": "test",
            "data_path": "tests/data/core/datastores/multiple_item_2.csv",
            "has_header": True,
            "skipinitialspace": True,
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 1

        items = datastore.load_data()
        assert len(items) == 3

        assert {"id": "1", "content": "This is sample content 1"} == items[0]
        assert {"id": "2", "content": "This is sample content 2"} == items[1]
        assert {"id": "3", "content": "This is sample content 3"} == items[2]

    def test_multiple_data_formats(self):
        primary_cfg = {
            "store_name": "knowledge",
            "data_path": "tests/data/core/datastores/single_*",
            "data_formats": ["json", "jsonl"],
        }

        datastore = DefaultDatastore(
            type="default",
            **primary_cfg,
        )

        iters = datastore.load_iterators()
        assert len(iters) == 3

        items = datastore.load_data()
        assert len(items) == 3
