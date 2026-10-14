"""Local evidence API; import FastAPI only when the optional demo is used."""


def create_app(*args, **kwargs):
    from trialgrid.api.app import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
