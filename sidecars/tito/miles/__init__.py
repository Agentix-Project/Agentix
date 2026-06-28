"""Compatibility namespace for vendored Miles TITO modules."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
