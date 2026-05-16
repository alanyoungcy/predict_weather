import json
from unittest.mock import patch

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'api' / 'app.py'
spec = spec_from_file_location('mongo_test_endpoint', MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_evaluate_request_returns_401_when_token_required_and_missing(monkeypatch):
    monkeypatch.setenv('MONGO_TEST_TOKEN', 'secret')
    payload, status = module._evaluate_request({}, {})
    assert status == 401
    assert payload['message'] == 'unauthorized'


def test_evaluate_request_returns_payload_when_authorized(monkeypatch):
    monkeypatch.setenv('MONGO_TEST_TOKEN', 'secret')
    with patch.dict(module.os.environ, {'MONGODB_URI': 'mongodb+srv://example'}):
        with patch.object(module, 'MongoClient') as client_cls:
            client = client_cls.return_value
            client.admin.command.side_effect = [{'ok': 1}, {'version': '8.0.0'}]
            client.topology_description.topology_type_name = 'ReplicaSetWithPrimary'
            client.topology_description.server_descriptions.return_value = {}
            payload, status = module._evaluate_request({'x-mongo-test-token': 'secret'}, {})
    assert status == 200
    assert payload['ok'] is True
    assert payload['message'] == 'ok'
