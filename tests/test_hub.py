"""Model names on the command line: folders, Hugging Face model IDs and the served model name."""

import pytest

from seventy import hub


@pytest.mark.parametrize("name, ok", [
    ("QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", True),
    ("incoai/Qwen3.8-27B-DFlash2", True),
    ("owner/name_1.0", True),
    ("name-only", False),
    ("/abs/path", False),
    ("./models/x", False),
    ("a/b/c", False),
    ("~/models", False),
    ("owner/na..me", False),
    ("owner/na--me", False),
    ("owner/", False),
    ("owner/name with space", False),
])
def test_is_model_id(name, ok):
    assert hub.is_model_id(name) is ok


def test_resolve_keeps_a_folder_and_does_not_call_the_hub(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("called the Hub"))
    folder = tmp_path / "owner" / "name"
    folder.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert hub.resolve(str(folder)) == str(folder)
    assert hub.resolve("owner/name") == "owner/name"


def test_resolve_gets_a_model_id_from_snapshot_download(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    snap = tmp_path / "hub" / "models--owner--name" / "snapshots" / "abc123"
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **k: calls.append(repo) or str(snap))
    monkeypatch.chdir(tmp_path)
    assert hub.resolve("owner/name") == str(snap)
    assert calls == ["owner/name"]


def test_resolve_errors(tmp_path, monkeypatch):
    import huggingface_hub

    def fail(repo, **k):
        raise OSError("offline and not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(hub.ModelNotFound, match="drafter owner/name.*offline"):
        hub.resolve("owner/name", "drafter")
    with pytest.raises(hub.ModelNotFound, match="not a folder"):
        hub.resolve(str(tmp_path / "missing"))


def test_served_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    snap = tmp_path / "hub" / "models--QUASAR-QAT--Qwen3.8-27B-QUASAR-NVFP4" / "snapshots" / "0123abcd"
    assert hub.served_name(str(snap), str(snap)) == "Qwen3.8-27B-QUASAR-NVFP4"
    assert hub.served_name("QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4", str(snap)) == "Qwen3.8-27B-QUASAR-NVFP4"
    assert hub.served_name("owner/name", str(tmp_path / "elsewhere")) == "name"
    local = tmp_path / "models" / "qwen-27b"
    assert hub.served_name(str(local) + "/", str(local) + "/") == "qwen-27b"
