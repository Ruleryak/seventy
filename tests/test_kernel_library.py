"""The kernel library loads on the Linux systems of its wheel tag (manylinux_2_28): glibc 2.28 or later."""

import re
import shutil
import subprocess

import pytest

from seventy.ops import _lib


@pytest.fixture(scope="module")
def library():
    path = _lib.path()
    if path is None or not path.is_file():
        pytest.skip("the kernel library is not built")
    if shutil.which("objdump") is None:
        pytest.skip("needs objdump (binutils)")
    return path


def test_needs_glibc_2_28_at_most(library):
    out = subprocess.run(["objdump", "-T", str(library)], capture_output=True, text=True, check=True).stdout
    versions = {tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", out)}
    assert versions and max(versions) <= (2, 28), f"needs glibc {'.'.join(map(str, max(versions)))}"


def test_uses_the_shared_cuda_runtime(library):
    out = subprocess.run(["objdump", "-p", str(library)], capture_output=True, text=True, check=True).stdout
    needed = re.findall(r"NEEDED\s+(\S+)", out)
    assert "libcudart.so.12" in needed
    assert not any(n.startswith("libcuda.so") for n in needed)
