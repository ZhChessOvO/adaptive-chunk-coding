"""Real two-layer rANS using UF's installed CPU coder, without changing it.

HT-S scales depend only on received z and the base temporal reference. The
Gaussian-bin CDFs here are a DECLARED approximate model, not the original UF
integer CDF. E uses p(fine | received coarse bin, scale); groups of equal coarse
bin work around the native coder's 8-bit CDF index. Group order follows B and
requires no transmitted mask. Packet/group overhead is included in real bytes.
"""
from functools import lru_cache
import math
import struct

import numpy as np
from scipy.special import ndtr

from routervc.latent.split import validate_width, restore_symbols


def scale_indexes(scales):
    s = np.asarray(scales, dtype=np.float64)
    if not np.isfinite(s).all(): raise ValueError('nonfinite entropy scales')
    return np.floor((np.log(np.clip(s, .11, 16.))-math.log(.11)) /
                    (math.log(16./.11)/127.)).clip(0,127).astype(np.uint8)


def _masses(values, sigma):
    # Both endpoints in the negative tail avoid catastrophic 1-CDF subtraction.
    a = np.abs(values)
    return np.maximum(ndtr((.5-a)/sigma)-ndtr((-.5-a)/sigma), 1e-30)


@lru_cache(maxsize=256)
def tables(width, coarse_bin=None):
    import torch  # Load native library dependencies before the CPU extension.
    from MLCodec_extensions_cpp import pmf_to_quantized_cdf
    validate_width(width)
    if width < 3: raise ValueError('stream profile requires width >= 3')
    scales=np.exp(np.linspace(math.log(.11),math.log(16.),128))
    half=width//2
    radius=int(max(abs((-128+half)//width), abs((127+half)//width)))
    bound=radius if coarse_bin is None else half
    coordinates=np.arange(-bound,bound+1)
    # Native rANS reorders signed integers to 0,+1,-1,+2,-2,... then escape.
    order=np.array([bound]+[j for i in range(1,bound+1) for j in (bound+i,bound-i)])
    cdfs=[]
    for sigma in scales:
        if coarse_bin is None:
            pmf=np.array([_masses(np.arange(max(-128,b*width-half),min(127,b*width+half)+1),sigma).sum()
                          for b in coordinates])
        else:
            values=coarse_bin*width+coordinates
            pmf=_masses(values,sigma)
            pmf[(values < -128)|(values > 127)]=0
        if pmf.sum() <= 0: raise ValueError('empty conditional distribution')
        pmf=pmf/pmf.sum()
        probs=np.r_[pmf[order]*(1.-1e-9),1e-9]
        cdfs.append(pmf_to_quantized_cdf(probs.tolist()))
    cdf=np.array(cdfs,dtype=np.int32)
    lengths=np.full(128,cdf.shape[1],dtype=np.int32)
    if cdf.shape[1]-2 > 127: raise ValueError('native rANS CDF alphabet overflow')
    return cdf,lengths


def _coder(cdf_info):
    import torch
    from MLCodec_extensions_cpp import RansEncoder,RansDecoder
    encoder,decoder=RansEncoder(),RansDecoder()
    for coder in (encoder,decoder):
        coder.set_entropy_coder_parallel(1)
        coder.set_cdf(*cdf_info,1)
    return encoder,decoder


def encode_values(values,indexes,cdf_info):
    values=np.asarray(values)
    indexes=np.asarray(indexes,dtype=np.uint8)
    if values.size!=indexes.size or values.dtype.kind not in 'iu': raise ValueError('invalid symbols/indexes')
    if values.size==0 or values.min() < -128 or values.max() > 127: raise ValueError('invalid signed alphabet')
    packed=((values.astype(np.int32).ravel() << 8)|indexes.ravel()).astype(np.int16)
    encoder,_=_coder(cdf_info)
    encoder.reset();encoder.encode_y(packed);encoder.flush()
    return encoder.get_encoded_stream().tobytes()


def decode_values(data,indexes,cdf_info):
    indexes=np.asarray(indexes,dtype=np.uint8)
    if len(data)<4 or indexes.size==0: raise ValueError('empty or truncated rANS')
    _,decoder=_coder(cdf_info)
    decoder.set_stream(np.frombuffer(data,dtype=np.uint8))
    decoder.decode_y(indexes.ravel().copy())
    return decoder.get_decoded_tensor().astype(np.int16).reshape(indexes.shape)


def encode_coarse(coarse,scales,width):
    return encode_values(coarse,scale_indexes(scales),tables(width))


def decode_coarse(data,scales,width):
    return decode_values(data,scale_indexes(scales),tables(width))


def encode_fine(coarse,fine,scales,width):
    restore_symbols(coarse,fine,width)  # Validate before any narrowing casts.
    indexes=scale_indexes(scales)
    out=bytearray()
    for value in np.unique(coarse):
        mask=coarse==value
        part=encode_values(fine[mask],indexes[mask],tables(width,int(value)))
        out+=struct.pack('<I',len(part))+part
    return bytes(out)


def decode_fine(data,coarse,scales,width):
    indexes=scale_indexes(scales);out=np.empty_like(coarse,dtype=np.int16);cursor=0
    for value in np.unique(coarse):
        if len(data)-cursor < 4: raise ValueError('truncated refinement group header')
        size=struct.unpack_from('<I',data,cursor)[0];cursor+=4
        if size < 4 or size > len(data)-cursor: raise ValueError('truncated refinement group')
        mask=coarse==value
        out[mask]=decode_values(data[cursor:cursor+size],indexes[mask],tables(width,int(value)))
        cursor+=size
    if cursor!=len(data): raise ValueError('trailing refinement bytes')
    restore_symbols(coarse,out,width)
    return out


def encode_z(z,cdf_info,qp):
    import torch
    from MLCodec_extensions_cpp import RansEncoder
    if np.any(z < -128) or np.any(z > 127) or np.any(z!=np.round(z)):
        raise ValueError('hyperlatent outside native signed alphabet')
    encoder=RansEncoder();encoder.set_entropy_coder_parallel(1);encoder.set_cdf(*cdf_info,0)
    encoder.reset()
    # Native z entropy order is NHWC, not tensor's logical NCHW.
    encoder.encode_z(z.transpose(0,2,3,1).astype(np.int8).ravel(),qp*128,128)
    encoder.flush()
    return encoder.get_encoded_stream().tobytes()


def decode_z(data,shape,cdf_info,qp):
    import torch
    from MLCodec_extensions_cpp import RansDecoder
    if len(shape)!=4 or shape[:2]!=(1,128) or min(shape)<1 or len(data)<4:
        raise ValueError('invalid z shape/stream')
    decoder=RansDecoder();decoder.set_entropy_coder_parallel(1);decoder.set_cdf(*cdf_info,0)
    decoder.set_stream(np.frombuffer(data,dtype=np.uint8))
    decoder.decode_z(int(np.prod(shape)),qp*128,128)
    z=decoder.get_decoded_tensor().reshape(shape[0],shape[2],shape[3],shape[1])
    return z.transpose(0,3,1,2).astype(np.float16)
