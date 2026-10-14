try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:  # pragma: no cover - dependency is optional at import time
    load_dataset = None
    _datasets_import_error = exc
else:
    _datasets_import_error = None


def main():
    if load_dataset is None:
        raise RuntimeError(
            "The 'datasets' package is required to download the CORD dataset. "
            "Install it with: pip install datasets"
        ) from _datasets_import_error

    dataset = load_dataset("naver-clova-ix/cord-v2")
    print(dataset)


if __name__ == "__main__":
    main()
