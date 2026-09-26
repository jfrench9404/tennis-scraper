"""Download a pinned GridTrackNet checkpoint and export its inference graph.

No remote Python, pickle, or serialized executable model is loaded. Keras HDF5
numeric weights are mapped to standard ONNX operators. Architecture attribution:
Vincent Korpelshoek's GridTrackNet, MIT; see THIRD-PARTY-NOTICES.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '0764162b73fb64d440fd9e6c363d592965400799'
URL = f'https://raw.githubusercontent.com/VKorpelshoek/GridTrackNet/{REVISION}/model_weights.h5'
WEIGHTS_SHA256 = 'ac93a1f074b5292c6a06474db1fb8a2ff91553816eb337d6062b28535056ee42'
DEFAULT_MODEL = Path(__file__).resolve().parents[1]/'models/gridtracknet/gridtracknet.onnx'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def export_graph(weights, output):
    import h5py
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    if file_hash(weights) != WEIGHTS_SHA256:
        raise ValueError('Checkpoint hash mismatch. This exporter supports only the pinned public weights.')
    nodes=[]; constants=[]; current='frames'
    def constant(name,array):
        constants.append(numpy_helper.from_array(np.asarray(array,np.float32),name))
        return name
    with h5py.File(weights,'r') as model:
        for index in range(13):
            suffix=f'_{index}' if index else ''
            name='conv2d'+suffix
            group=model[name][name]
            kernel=np.asarray(group['kernel:0']).transpose(3,2,0,1)
            bias=np.asarray(group['bias:0'])
            output_name=f'conv_{index}'
            nodes.append(helper.make_node('Conv',[current,constant(name+'_kernel',kernel),constant(name+'_bias',bias)],
                                           [output_name],pads=[1,1,1,1],kernel_shape=[3,3]))
            current=output_name
            if index==12:
                nodes.append(helper.make_node('Sigmoid',[current],['grid']))
                break
            nodes.append(helper.make_node('Relu',[current],[f'relu_{index}']))
            name='batch_normalization'+suffix
            group=model[name][name]
            # Upstream uses default axis=-1 on NCHW tensors: normalization is
            # per image column, NOT per channel. Preserve the trained semantics.
            gamma=np.asarray(group['gamma:0'])
            beta=np.asarray(group['beta:0'])
            mean=np.asarray(group['moving_mean:0'])
            variance=np.asarray(group['moving_variance:0'])
            scale=gamma/np.sqrt(variance+.001)
            shift=beta-mean*scale
            nodes.append(helper.make_node('Mul',[f'relu_{index}',constant(name+'_scale',scale.reshape(1,1,1,-1))],[f'scale_{index}']))
            current=f'bn_{index}'
            nodes.append(helper.make_node('Add',[f'scale_{index}',constant(name+'_shift',shift.reshape(1,1,1,-1))],[current]))
            if index in (1,3,5,8):
                nodes.append(helper.make_node('MaxPool',[current],[f'pool_{index}'],kernel_shape=[2,2],strides=[2,2]))
                current=f'pool_{index}'
    graph=helper.make_graph(nodes,'GridTrackNet-five-frame',
                            [helper.make_tensor_value_info('frames',TensorProto.FLOAT,[1,15,432,768])],
                            [helper.make_tensor_value_info('grid',TensorProto.FLOAT,[1,15,27,48])],constants)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],producer_name='tennis-vision-mvp')
    model.ir_version=10
    helper.set_model_props(model,{'source_revision':REVISION,'weights_sha256':WEIGHTS_SHA256,
                                  'input_colour':'RGB','sequence_frames':'5','license':'MIT'})
    onnx.checker.check_model(model)
    onnx.save_model(model,str(output))


def prepare(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    weights=directory/'model_weights.h5'
    if not weights.exists():
        temporary=directory/'model_weights.h5.download'
        print('Downloading pinned tennis checkpoint (35.7 MB)...',flush=True)
        urllib.request.urlretrieve(URL,temporary)
        if file_hash(temporary)!=WEIGHTS_SHA256:
            raise ValueError('Downloaded checkpoint failed SHA-256 verification; not loading it.')
        temporary.replace(weights)
    if file_hash(weights)!=WEIGHTS_SHA256:
        raise ValueError('Existing checkpoint has an unexpected hash; left unchanged.')
    output=directory/'gridtracknet.onnx'
    if output.exists():
        manifest=directory/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()).get('onnx_sha256')==file_hash(output):
            print(f'Existing verified model: {output}')
            return output
        raise ValueError('Existing ONNX model is unverified; choose a new output directory.')
    export_graph(weights,output)
    manifest={'source':'https://github.com/VKorpelshoek/GridTrackNet','revision':REVISION,'license':'MIT',
              'weights_sha256':WEIGHTS_SHA256,'onnx_sha256':file_hash(output),'input':'5 RGB frames, 768x432, divided by 255',
              'note':'Pretrained public tennis model; not fine-tuned on this camera. Scores are not calibrated accuracy.'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Ready: {output}',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=DEFAULT_MODEL.parent)
    prepare(parser.parse_args().directory)
