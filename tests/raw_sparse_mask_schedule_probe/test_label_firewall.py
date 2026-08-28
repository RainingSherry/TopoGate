import inspect

from scripts.raw_sparse_mask_schedule_probe import model, raw_adapter


def test_fit_path_has_no_label_loader_reference():
    source = inspect.getsource(model.fit_autoencoder)
    assert "load_labels_after_fit" not in source
    assert "labels" not in source.lower()


def test_label_loader_is_explicit_post_fit_entrypoint():
    assert "after_fit" in raw_adapter.load_labels_after_fit.__name__
