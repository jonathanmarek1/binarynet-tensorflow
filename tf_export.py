import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np
import copy

# attempt to evaluate a tensorflow variable
# (needed because batch norm has non-fetchable variables)
def eval(var):
    op = var.op
    if op.type == 'Identity':
        val = op.inputs[0].eval(feed_dict={'is_training:0':False})
    elif op.type == 'Switch':
        val = op.inputs[0].eval(feed_dict={'is_training:0':False})
    elif op.type == 'Const':
        val = tensor_util.MakeNdarray(op.get_attr('value'))
    else:
        val = var.eval(feed_dict={'is_training:0':False})
    return val

# layer tracking when parsing graph
class Layer(object):
    def __init__(self):
        self.m = np.ones(1)
        self.b = np.zeros(1)
        self.act = 'none'
        self.pool = None
        self.binary = False
        self.relu_fix = None

    def mul(self, m):
        self.m = self.m * m

    def add(self, b):
        self.b = self.b + self.m * b

    def activate(self, act):
        if act == 'relu' and self.act == 'bin': # xnornet workaround
            assert(self.relu_fix is None)
            self.relu_fix = -self.b / self.m < 0.0
            return
        assert(np.array_equal(self.m, np.ones(1)))
        assert(np.array_equal(self.b, np.zeros(1)))
        self.act = act;

    def max_pool(self, pool):
        assert(self.pool == None)
        self.pool = pool
        self.pool_xor = self.m < 0.0 # for xnornet workaround

    def finish(self, w, strides, padding, binary, num_output):
        print('finish')
        self.w = w
        self.strides = strides
        self.padding = padding
        self.binary = binary
        self.num_output = num_output
#
def clipto(x, type):
    info = np.iinfo(np.int32)
    return np.clip(x, info.min, info.max)

#
def reorder(x, group_size, pairing):
    assert(x.shape[2] % pairing == 0)

    x0 = None
    if x.shape[3] % group_size:
        off = x.shape[3] // group_size * group_size;
        x0 = x[:,:,:,off:]
        x = x[:,:,:,:off]

        oldshape = x0.shape
        x0 = np.reshape(x0, (*x0.shape[:2], x0.shape[2] // pairing, pairing, 1, x0.shape[3]))
        x0 = np.transpose(x0, (0, 1, 4, 2, 5, 3))
        x0 = np.reshape(x0, oldshape)

    oldshape = x.shape
    x = np.reshape(x, (*x.shape[:2], x.shape[2] // pairing, pairing, \
                                     x.shape[3] // group_size, group_size))
    x = np.transpose(x, (0, 1, 4, 2, 5, 3))
    x = np.reshape(x, oldshape)

    x = np.reshape(x, (*x.shape[:2], -1))

    if x0 is not None:
        x0 = np.reshape(x0, (*x0.shape[:2], -1))
        x = np.concatenate([x, x0], axis=2)
    return x

"""
Notes
- some optimizations missing
- only supports convolutional networks
"""

def export(output, input, prefix, quantize):

    weight_data = bytearray()
    layer_param = []

    processed = []
    layers = []
    input_scale = None
    input_offset = None
    input_type = 'FLOAT'

    var = [(output, Layer())]

    binary = False

    while var:
        _var = var
        var = []
        for (v, layer) in _var:
            if v in processed:
                print('TODO: VARIABLE IS REUSED')
                continue
            processed += [v]

            op = v.op

            if op.type == 'Identity':
                assert(len(op.inputs) == 1 and len(op.outputs) == 1)
            elif op.type == 'Merge':
                assert(v == op.outputs[0])
                var += [(i, copy.copy(layer)) for i in op.inputs]
                continue
            elif op.type == 'Switch':
                assert(len(op.inputs) == 2 and len(op.outputs) == 2)
                pred = eval(op.inputs[1])
                if v == op.outputs[pred*1]:
                    var += [(op.inputs[0], layer)]
                continue
            elif op.type == 'FusedBatchNorm':
                assert(len(op.inputs) == 5 and len(op.outputs) == 5)

                epsilon = op.get_attr('epsilon')
                scale = eval(op.inputs[1])
                offset = eval(op.inputs[2])
                mean = eval(op.inputs[3])
                variance = eval(op.inputs[4])

                if mean.size == 0:
                    mean = np.zeros(1)
                if variance.size == 0:
                    variance = np.ones(1)

                m = scale / np.sqrt(variance + epsilon)
                b = -mean * m + offset

                layer.add(b)
                layer.mul(m)
            elif op.type == 'Add' or op.type == 'Sub' or op.type == 'Mul':
                assert(len(op.inputs) == 2 and len(op.outputs) == 1)

                # detect binary activation (hacky)
                if op.type == 'Add' and op.inputs[1].op.type == 'StopGradient':
                    x = op.inputs[0].op
                    y = x.inputs[0].op
                    assert(x.type == 'Maximum')
                    assert(y.type == 'Minimum')
                    assert(layer.m == np.ones(1) and layer.b == np.zeros(1))

                    layer.activate('bin')
                    op = y
                elif op.type == 'Add':
                    layer.add(op.inputs[1].eval())
                elif op.type == 'Sub':
                    layer.add(-op.inputs[1].eval())
                else:
                    layer.mul(op.inputs[1].eval())
            elif op.type == 'Placeholder':
                assert(len(op.inputs) == 0 and len(op.outputs) == 1)
                assert(v == input)
                print('input parameters', layer.m, layer.b)
                input_scale = layer.m
                input_offset = layer.b
                if layer.act == 'bin':
                    binary = True
                    input_type = 'BINARY'
                else:
                    assert(layer.act == 'none')
                continue
            elif op.type == 'MaxPool':
                assert(len(op.inputs) == 1 and len(op.outputs) == 1)
                assert(op.get_attr('padding') == b'VALID')
                layer.max_pool((op.get_attr('ksize'), op.get_attr('strides')))
            elif op.type == 'Conv2D':
                assert(len(op.inputs) == 2 and len(op.outputs) == 1)
                op2 = op.inputs[0].op
                inp = op.inputs[0]

                padding = [0, 0]
                W = op.inputs[1].eval()

                if op2.type == 'Pad':
                    assert(len(op2.inputs) == 2 and len(op2.outputs) == 1)
                    #assert(op2.get_attr('mode') == 'CONSTANT')
                    inp = op2.inputs[0]
                    pad = op2.inputs[1].eval()
                    assert(np.all(pad[0] == 0) and np.all(pad[3] == 0))
                    assert(pad[1][0] == pad[1][1] and pad[2][0] == pad[2][1])
                    padding = [pad[1][0], pad[2][0]]

                if op.get_attr('padding') == b'SAME':
                    padding[0] += W.shape[0] // 2
                    padding[1] += W.shape[1] // 2

                # detect binary weights (TODO)
                bw = False
                if op.inputs[1].op.type == 'Add':
                    bw = True

                print(padding, bw, W.shape)

                stride = op.get_attr('strides')
                assert(stride[0] == 1 and stride[3] == 1)
                strides = [stride[1], stride[2]]

                layer.finish(W, strides, padding, bw, np.prod([x.value for x in op.outputs[0].shape[1:]]))
                layers += [layer]
                var += [(inp, Layer())]
                continue
            elif op.type == 'Relu':
                assert(len(op.inputs) == 1 and len(op.outputs) == 1)
                layer.activate('relu')
            else:
                print('Unknown operation:', op.type)
                assert(False)
            var += [(op.inputs[0], layer)]

    weight_data = bytearray()
    layers = layers[::-1]
    code = []
    codew = []

    tmp_size = 0 # TODO initialize to size of input

    for i, layer in enumerate(layers):
        shape = layer.w.shape

        # update required size of temp memory
        k = layer.num_output
        if layer.act == 'bin':
            k = (layer.num_output + 7) // 8
        else:
            k = layer.num_output * 4
        tmp_size = max(tmp_size, k)

        #
        in_size = np.prod(layer.w.shape[:-1])

        if layer.binary:
            layer.m = layer.m * np.mean(np.abs(layer.w), axis=(0, 1, 2))
            layer.wb = layer.w < 0.0
            layer.w = None
        else:
            layer.w *= layer.m
            layer.m = None

        xnornet_fix = None
        if layer.act == 'bin' and layer.binary:
            # xnornet negative scaling workarounds
            if layer.pool is not None:
                xnornet_fix = layer.pool_xor
                assert(np.all((layer.m < 0.0) == layer.pool_xor))
            else:
                xnornet_fix = layer.m < 0.0

            if not np.any(xnornet_fix):
                xnornet_fix = None
        elif layer.act == 'bin' and layer.pool is not None and np.any(layer.pool_xor):
            print('fix', layer.pool_xor[0])
            layer.w *= np.where(layer.pool_xor, -1.0, 1.0)
            layer.b *= np.where(layer.pool_xor, -1.0, 1.0)
            xnornet_fix = layer.pool_xor

        # extra parameters
        if binary:
            assert(layer.binary)
            if layer.act == 'bin':
                k = -layer.b / layer.m
                k = (in_size - k) / 2.0
                k = np.floor(np.clip(k, 0, np.iinfo(np.uint16).max)).astype(np.uint16)
                name = 'bin'
            else:
                k = np.concatenate([layer.m, layer.b]).astype(np.float32)
                name = 'bin_float'
        else:
            if layer.binary:
                assert(layer.act != 'bin') # TODO
                k = np.concatenate([layer.m, layer.b]).astype(np.float32)
                name =  'bin_float'
            else:
                if layer.act == 'bin' and layer.relu_fix is not None:
                    layer.b = np.where(layer.relu_fix, np.inf, layer.b)
                    layer.relu_fix = None

                if quantize:
                    _min = np.min(layer.w, axis=(0,1,2))
                    _max = np.max(layer.w, axis=(0,1,2))
                    m = 255.0 / (_max - _min)

                    # used floored min so integer arithmetic can be used
                    layer_min = np.floor(_min * m)
                    _min = layer_min * _max / (255.0 + layer_min)
                    m = 255.0 / (_max - _min)

                    layer.w = np.round(layer.w * m - layer_min) - 128.0
                    layer.w = np.clip(layer.w, -128.0, 127.0) # shouldnt be necessary

                    k = np.concatenate([layer.b, 1.0 / m, layer_min + 128.0, np.sum(layer.w, axis=(0,1,2))])
                else:
                    k = layer.b

                k = k.astype(np.float32)
                name = 'int8' if quantize else 'float'

        assert(layer.relu_fix is None)

        # reordering
        # we can gain some performance by ordering the weights in a way
        # specific to the implementation
        # currently set for armv7a implementation
        group_size = 32 # 96
        if layer.binary:
            if binary:
                group_size = 64 # 128
                assert(layer.wb.shape[3] % group_size == 0)
                wd = np.packbits(reorder(layer.wb, group_size, 8))
            else:
                # float input with binary out
                assert(layer.wb.shape[3] % group_size == 0)
                assert(layer.wb.shape[2] % 2 == 0)
                if quantize:
                    assert(group_size % 8 == 0)

                    x = layer.wb
                    x = np.reshape(x, (*x.shape[:2], x.shape[2] // 2, 2, \
                                   x.shape[3] // group_size, group_size // 8, 8))
                    x = np.transpose(x, (0, 1, 4, 2, 5, 3, 6))
                    wd = np.packbits(x)
                else:
                    wd = np.packbits(reorder(layer.wb, group_size, 2))
        else:
            if quantize:
                wd = reorder(layer.w, group_size, 1).astype(np.int8)
            else:
                wd = reorder(layer.w, group_size, 1).astype(np.float32)


        #code for this layer
        if quantize and not binary:
            code += ['x = quantize(x, arg->quant_param, %i, %i);' % (1 if layer.binary else 0, 1 if i == 0 else 0)]

        code += ['x = conv2d(x, (tensor) {4, {%i, %i, %i, %i}, w->layer%i.w, .type=%s}, (tensor) {2, {1, %i}, w->layer%i.b}, %i, %i, %i, %i, %s, %s, arg->sync);' % (*shape, i, 'BINARY' if layer.binary else 'INT8' if quantize else 'FLOAT', shape[3], i, layer.strides[0], layer.strides[1], layer.padding[0], layer.padding[1], 'ACTIVE_' + layer.act.upper(), 'arg->quant_param' if quantize and not binary else '0')]

        if layer.pool is not None:
            code += ['x = maxpool(x, %i, %i, %i, %i, %s, arg->sync);' % (layer.pool[0][1], layer.pool[0][2], layer.pool[1][1], layer.pool[1][2], ('w->xor%i' % i) if xnornet_fix is not None else '0')]
        elif xnornet_fix is not None:
            code += ['x = xnornet_fix(x, w->xor%i);' % i]

        codew += ['w_%s(%i, %i) layer%i;' % (name, np.prod(shape[:-1]), shape[-1], i)]


        weight_data += wd.tobytes() + k.tobytes()

        if xnornet_fix is not None:
            # pad to 16byte multiple
            size = ((shape[-1] - 1) // 128) + 1
            xnornet_fix = np.resize(xnornet_fix, size * 128)
            codew += ['uint8_t xor%i[%i/8];' % (i, size*128)]
            weight_data += np.packbits(xnornet_fix).tobytes()

        print('layer', i, name)
        binary = layer.act == 'bin'

    print(len(weight_data))

    assert(tmp_size % 16 == 0)

    code = [x+' wait(arg, %i);'%(i+1) for (i, x) in enumerate(code)]

    c = '#define FUNCTION_NAME %s\n#include <c_ops.h>\n' % prefix
    c += 'struct weights {\n' + '\n'.join(codew) + '};\n'
    c += '_Static_assert(sizeof(struct weights) == %i, "");\n' % len(weight_data)

    c += 'static void* worker(void *_arg) {\n'
    c += 'struct thread_arg *arg = _arg;;\n'
    c += 'struct weights *w = arg->weights;\n'
    c += '#ifdef PRINT_TIME\nuint64_t t0 = get_time(), t1;\n#endif\n'
    c += 'tensor x = (tensor) {3, {%s}, __builtin_assume_aligned(arg->in, 16), {__builtin_assume_aligned(arg->tmp, 16), __builtin_assume_aligned(arg->tmp, 16) + %i}, .type=%s};\n' % (', '.join([str(x) for x in input.shape[1:]]), tmp_size, input_type)
    c += ' TIME();\n'.join(code)
    c += ' TIME();\nreturn x.data;\n}\n'
    #output.shape[1].value

    h = '#include <stdint.h>\n'
    h += '#define %s_size %i\n' % (prefix, len(weight_data))
    h += '#define %s_tmp_size (2*%i+16)\n' % (prefix, tmp_size)
    h += 'void* %s(void *in, void *weights, void *tmp);\n' % prefix

    f = open(prefix + '_weights', 'wb')
    f.write(weight_data)
    f.close()

    f = open(prefix + '.c', 'w')
    f.write(c)
    f.close()

    f = open(prefix + '.h', 'w')
    f.write(h)
    f.close()
