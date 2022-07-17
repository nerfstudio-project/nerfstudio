"""
Test tensor dataclass
"""
import pytest
import torch

from pyrad.utils.tensor_dataclass import tensordataclass


# pylint: disable=no-member,not-an-iterable,too-few-public-methods
@tensordataclass
class TestNestedClass:
    """Dummy dataclass"""

    x: torch.Tensor


# pylint: disable=no-member,not-an-iterable,too-few-public-methods
@tensordataclass
class TestTensorDataclass:
    """Dummy dataclass"""

    a: torch.Tensor
    b: torch.Tensor
    c: TestNestedClass = None


def test_init():
    """Test that dataclass is properly initialized"""

    @tensordataclass
    class Dummy:
        """Dummy dataclass"""

        dummy_vals: torch.Tensor = None

    dummy = Dummy(dummy_vals=torch.ones(1))
    with pytest.raises(ValueError):
        dummy = Dummy()


def test_broadcasting():
    """Test broadcasting during init"""

    a = torch.ones((4, 6, 3))
    b = torch.ones((6, 2))
    tensor_dataclass = TestTensorDataclass(a=a, b=b)
    assert tensor_dataclass.b.shape == (4, 6, 2)

    a = torch.ones((4, 6, 3))
    b = torch.ones((2))
    tensor_dataclass = TestTensorDataclass(a=a, b=b)
    assert tensor_dataclass.b.shape == (4, 6, 2)

    # Invalid broadcasting
    a = torch.ones((4, 6, 3))
    b = torch.ones((3, 2))
    with pytest.raises(RuntimeError):
        tensor_dataclass = TestTensorDataclass(a=a, b=b)


def test_tensor_ops():
    """Test tensor operations"""

    a = torch.ones((4, 6, 3))
    b = torch.ones((6, 2))
    tensor_dataclass = TestTensorDataclass(a=a, b=b)

    assert tensor_dataclass.shape == (4, 6)
    assert tensor_dataclass.a.shape == (4, 6, 3)
    assert tensor_dataclass.b.shape == (4, 6, 2)
    assert tensor_dataclass.size == 24
    assert tensor_dataclass.ndim == 2
    assert len(tensor_dataclass) == 4

    reshaped = tensor_dataclass.reshape((2, 12))
    assert reshaped.shape == (2, 12)
    assert reshaped.a.shape == (2, 12, 3)
    assert reshaped.b.shape == (2, 12, 2)

    flattened = tensor_dataclass.flatten()
    assert flattened.shape == (24,)
    assert flattened.a.shape == (24, 3)
    assert flattened.b.shape == (24, 2)
    assert flattened[0:4].shape == (4,)

    # Test indexing operations
    assert tensor_dataclass[:, 1].shape == (4,)
    assert tensor_dataclass[:, 1].a.shape == (4, 3)
    assert tensor_dataclass[:, 0:2].shape == (4, 2)
    assert tensor_dataclass[:, 0:2].a.shape == (4, 2, 3)
    assert tensor_dataclass[..., 1].shape == (4,)
    assert tensor_dataclass[..., 1].a.shape == (4, 3)
    assert tensor_dataclass[0].shape == (6,)
    assert tensor_dataclass[0].a.shape == (6, 3)
    assert tensor_dataclass[0, ...].shape == (6,)
    assert tensor_dataclass[0, ...].a.shape == (6, 3)

    tensor_dataclass = TestTensorDataclass(a=torch.ones((2, 3, 4, 5)), b=torch.ones((4, 5)))
    assert tensor_dataclass[0, ...].shape == (3, 4)
    assert tensor_dataclass[0, ...].a.shape == (3, 4, 5)
    assert tensor_dataclass[0, ..., 0].shape == (3,)
    assert tensor_dataclass[0, ..., 0].a.shape == (3, 5)
    assert tensor_dataclass[..., 0].shape == (2, 3)
    assert tensor_dataclass[..., 0].a.shape == (2, 3, 5)

    mask = torch.rand(size=(2, 3)) > 0.5
    assert tensor_dataclass[mask].ndim == 2


def test_nested_class():
    """Test nested TensorDataclasses"""

    a = torch.ones((4, 6, 3))
    b = torch.ones((6, 2))
    c = TestNestedClass(x=torch.ones(6, 5))
    tensor_dataclass = TestTensorDataclass(a=a, b=b, c=c)

    assert tensor_dataclass.shape == (4, 6)
    assert tensor_dataclass.a.shape == (4, 6, 3)
    assert tensor_dataclass.b.shape == (4, 6, 2)
    assert tensor_dataclass.c.shape == (4, 6)
    assert tensor_dataclass.c.x.shape == (4, 6, 5)
    assert tensor_dataclass.size == 24
    assert tensor_dataclass.c.size == 24

    reshaped = tensor_dataclass.reshape((2, 12))
    assert reshaped.shape == (2, 12)
    assert reshaped.c.shape == (2, 12)
    assert reshaped.c.x.shape == (2, 12, 5)

    flattened = tensor_dataclass.flatten()
    assert flattened.c.shape == (24,)
    assert flattened.c.x.shape == (24, 5)

    # Test indexing operations
    assert tensor_dataclass[:, 1].c.shape == (4,)
    assert tensor_dataclass[:, 1].c.x.shape == (4, 5)

    mask = torch.rand(size=(4,)) > 0.5
    assert tensor_dataclass[mask].c.ndim == 2


def test_iter():
    """Test iterating over tensor dataclass"""
    tensor_dataclass = TestTensorDataclass(a=torch.ones((3, 4, 5)), b=torch.ones((3, 4, 5)))
    for batch in tensor_dataclass:
        assert batch.shape == (4,)
        assert batch.a.shape == (4, 5)
        assert batch.b.shape == (4, 5)


def test_packed():
    """Test packed tensor dataclass"""
    # a = torch.ones((2, 2, 3, 3), device="cuda:0")
    # b = torch.ones((2, 2, 3, 2), device="cuda:0")
    c = TestNestedClass(x=torch.rand((800, 8000, 1, 5), device="cuda:3"))
    # tensor_dataclass = TestTensorDataclass(a=a, b=b, c=c)
    # assert tensor_dataclass.is_packed() is False

    valid_mask = torch.rand((800, 8000, 1), device="cuda:3") > 0.5
    import tqdm

    import pyrad.cuda as pyrad_cuda

    packed_info = pyrad_cuda.pack(valid_mask)
    packed = c.pack(valid_mask)
    packed_data, packed_info = pyrad_cuda.pack_single_tensor(c.x, valid_mask)
    print(c.x[valid_mask].sum())
    print(packed_data.sum())

    # torch.cuda.synchronize()
    # for _ in tqdm.tqdm(range(1000)):
    #     packed_info = pyrad_cuda.pack(valid_mask)
    #     torch.cuda.synchronize()

    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(10)):
        packed_data, packed_info = pyrad_cuda.pack_single_tensor(c.x, valid_mask)
        _x = pyrad_cuda.unpack(packed_data, packed_info, c.x.shape)
        torch.cuda.synchronize()
    print(_x.sum())

    # # torch.cuda.synchronize()
    # # for _ in tqdm.tqdm(range(1000)):
    # #     packed = c.pack(valid_mask)
    # #     torch.cuda.synchronize()

    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(10)):
        packed = c.x[valid_mask]
        _x = torch.zeros_like(c.x)
        _x[valid_mask] = packed
        torch.cuda.synchronize()
    print(_x.sum())

    # print("_packed_info", packed._packed_info)
    # num_samples = packed._packed_info[-1, -2] + packed._packed_info[-1, -1]
    # assert packed.a.shape[0] == num_samples
    # assert packed.c.x.shape[0] == num_samples
    # assert packed.is_packed()
    # assert packed.c.is_packed()

    # unpacked = packed.unpack((4, 2, 6))
    # print(unpacked.a[0, 0])


if __name__ == "__main__":
    test_init()
    test_broadcasting()
    test_tensor_ops()
    test_iter()
    test_nested_class()
    test_packed()
