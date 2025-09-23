# Third Party
import pytest

# Local
from fms_dgt.core.dataloaders.default import (
    _ITER_INDEX,
    _ROW_INDEX,
    DatastoreDataloader,
)


class TestDatastoreDataloader:

    def test_single_iterate(self):
        test_iter = (i for i in range(10))
        dl = DatastoreDataloader(iterators=[test_iter])
        output = list(dl)
        assert output == list(range(10)), f"Expected full range, got {output}"

        # Confirm further next() raises StopIteration
        with pytest.raises(StopIteration):
            next(dl)

    def test_multiple_iterate(self):
        it1 = (i for i in range(10))
        it2 = (i for i in range(10, 20))
        dl = DatastoreDataloader(iterators=[it1, it2])
        output = list(dl)
        assert output == list(range(20)), f"Expected 0-19, got {output}"

    def test_resume_from_mid_first_iterator(self):
        it1 = (i for i in range(10))
        it2 = (i for i in range(10, 20))
        dl = DatastoreDataloader(iterators=[it1, it2])
        dl.set_state({_ITER_INDEX: 0, _ROW_INDEX: 5})

        output = list(dl)
        expected = list(range(5, 20))
        assert output == expected, f"Expected {expected}, got {output}"

    def test_resume_from_second_iterator(self):
        it1 = (i for i in range(10))
        it2 = (i for i in range(10, 20))
        dl = DatastoreDataloader(iterators=[it1, it2])
        dl.set_state({_ITER_INDEX: 1, _ROW_INDEX: 3})

        output = list(dl)
        expected = list(range(13, 20))
        assert output == expected, f"Expected {expected}, got {output}"

    def test_field_transformation(self):
        data = [
            {
                "id": i,
                "text": f"text-{i}",
                "meta": {"lang": "en", "region": "north america"},
            }
            for i in range(3)
        ]
        it = (item for item in data)
        fields = {
            "*": "*",
            "text": "content",
            "meta.lang": "language",
            "meta.region": "location.region",
        }

        dl = DatastoreDataloader(iterators=[it], fields=fields)

        output = list(dl)
        for original, transformed in zip(data, output):
            # All keys from original are present (wildcard "*")
            for k in original:
                assert k in transformed or k == "meta"  # nested field replaced
            # Extra renamed fields
            assert transformed["content"] == original["text"]
            assert transformed["language"] == original["meta"]["lang"]
            assert transformed["location"]["region"] == original["meta"]["region"]

    def test_missing_field_raises_keyerror(self):
        data = [{"id": 1, "foo": "bar"}]
        it = (item for item in data)
        fields = {"missing.key": "out"}

        dl = DatastoreDataloader(iterators=[it], fields=fields)

        with pytest.raises(KeyError):
            next(dl)
