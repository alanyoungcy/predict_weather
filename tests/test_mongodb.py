from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

from wt.storage.mongodb import _classify_mongo_error


def test_classify_mongo_replica_set_timeout() -> None:
    exc = ServerSelectionTimeoutError(
        "No replica set members found yet, Timeout: 10.0s, Topology Description: <TopologyDescription topology_type: ReplicaSetNoPrimary>"
    )
    message = _classify_mongo_error(exc)
    assert "no primary was selectable" in message.lower()
    assert "network access restrictions" in message.lower()


def test_classify_mongo_configuration_error() -> None:
    exc = ConfigurationError("Bad URI option")
    message = _classify_mongo_error(exc)
    assert message == "MongoDB configuration error: Bad URI option"


def test_classify_mongo_auth_error() -> None:
    exc = OperationFailure("bad auth")
    message = _classify_mongo_error(exc)
    assert message == "MongoDB authentication/authorization failed: bad auth"
