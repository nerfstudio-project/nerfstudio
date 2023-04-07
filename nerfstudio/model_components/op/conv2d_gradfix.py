import contextlib
import warnings

import torch
from torch import autograd
from torch.nn import functional as F

enabled = True
weight_gradients_disabled = False


@contextlib.contextmanager
def no_weight_gradients():
    global weight_gradients_disabled

    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
        ).apply(input, weight, bias)

    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    if could_use_op(input):
        return conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ).apply(input, weight, bias)

    return F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )


def could_use_op(input):
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False

    if input.device.type != "cuda":
        return False

    if any(torch.__version__.startswith(x) for x in ["1.7.", "1.8."]):
        return True

    warnings.warn(
        f"conv2d_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.conv2d()."
    )

    return False


def ensure_tuple(xs, ndim):
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim

    return xs


conv2d_gradfix_cache = dict()


def conv2d_gradfix(
    transpose, weight_shape, stride, padding, output_padding, dilation, groups
):
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = ensure_tuple(stride, ndim)
    padding = ensure_tuple(padding, ndim)
    output_padding = ensure_tuple(output_padding, ndim)
    dilation = ensure_tuple(dilation, ndim)

    key = (transpose, weight_shape, stride, padding, output_padding, dilation, groups)
    if key in conv2d_gradfix_cache:
        return conv2d_gradfix_cache[key]

    common_kwargs = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]

        return [
            input_shape[i + 2]
            - (output_shape[i + 2] - 1) * stride[i]
            - (1 - 2 * padding[i])
            - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    class Conv2d(autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            if not transpose:
                out = F.conv2d(input=input, weight=weight, bias=bias, **common_kwargs)

            else:
                out = F.conv_transpose2d(
                    input=input,
                    weight=weight,
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs,
                )

            ctx.save_for_backward(input, weight)

            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            grad_input, grad_weight, grad_bias = None, None, None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, weight, None)

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))

            return grad_input, grad_weight, grad_bias

    class Conv2dGradWeight(autograd.Function):
        @staticmethod
        def forward(ctx, grad_output, input):
            op = torch._C._jit_get_operation(
                "aten::cudnn_convolution_backward_weight"
                if not transpose
                else "aten::cudnn_convolution_transpose_backward_weight"
            )
            flags = [
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32,
            ]
            grad_weight = op(
                weight_shape,
                grad_output,
                input,
                padding,
                stride,
                dilation,
                groups,
                *flags,
            )
            ctx.save_for_backward(grad_output, input)

            return grad_weight

        @staticmethod
        def backward(ctx, grad_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_grad_output, grad_grad_input = None, None

            if ctx.needs_input_grad[0]:
                grad_grad_output = Conv2d.apply(input, grad_grad_weight, None)

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(
                    input_shape=input.shape, output_shape=grad_output.shape
                )
                grad_grad_input = conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs,
                ).apply(grad_output, grad_grad_weight, None)

            return grad_grad_output, grad_grad_input

    conv2d_gradfix_cache[key] = Conv2d

    return Conv2d
