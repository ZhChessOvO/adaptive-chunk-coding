"""Source-free single-P8 RVL1 receiver and native single-encoding sender.

This is the first two-layer diagnostic, NOT the final regional/G profile.
All four E/G combinations remain the final design goal. No G weights are loaded.
"""
from pathlib import Path
import hashlib
import io

import numpy as np

from routervc.latent import entropy, format as wire
from routervc.latent.split import split_symbols,restore_symbols,coarse_representatives
from demo.scalable_codec import BaseCodec,file_hash

REPO=Path(__file__).resolve().parents[2]
MODEL_I=REPO/'checkpoints/cvpr2026_image.pth.tar'
MODEL_P=REPO/'checkpoints/cvpr2026_video_hts.pth.tar'


def profile_hash():
    names=('__init__.py','split.py','entropy.py','format.py','codec.py','native.py','native_bridge.cpp')
    return hashlib.sha256(''.join(f'{n}:{file_hash(Path(__file__).with_name(n))}\n' for n in names).encode()).hexdigest()


class LatentCodec:
    def __init__(self):
        from demo.chunk_enhancement_codec import configure_torch
        from routervc.latent.native import load_bridge
        self.bridge,self.bindings=load_bridge()
        self.codec=BaseCodec(MODEL_I,MODEL_P)
        configure_torch()
        self.identities=(self.codec.models['model_i_sha256'],self.codec.models['model_p_sha256'],
                         self.bindings['native_library_sha256'],profile_hash())

    def rgb(self,result,h,w):
        from demo.stage_c_three_path_roi_probe import rgb_from_tensor
        return np.stack([rgb_from_tensor(result[f'frame{i}'],h,w) for i in range(8)])

    def tensor(self,array):
        import torch
        return torch.from_numpy(np.asarray(array).copy()).to(device=self.codec.device,dtype=torch.float16).contiguous(memory_format=torch.channels_last)

    def decode(self,data,*,allow_incomplete_tail=False):
        import torch
        from demo.stage_c_three_path_roi_probe import rgb_from_tensor
        parsed=wire.parse(data,allow_incomplete_tail=allow_incomplete_tail)
        if parsed.identities!=self.identities: raise ValueError('shared model/native/profile identity mismatch')
        h,w=parsed.height,parsed.image_width
        torch.cuda.set_stream(self.codec.stream)
        with torch.inference_mode():
            # Only the received native I initializes temporal reference. No caches.
            i=self.codec.i_net.decompress(parsed.intra,{'height':h,'width':w},32,parsed.i_parallel)['x_hat'].clone()
            self.codec.p_net.add_ref_feature_from_frame(i)
            proxy=self.codec.p_net.proxy;context=self.bridge.context(proxy)
            z=entropy.decode_z(parsed.z,(1,128,(h+63)//64,(w+63)//64),
                               self.codec.p_net.bit_estimator_z.get_cdf_info(),parsed.qp)
            zg=self.tensor(z)
            prior=self.bridge.prior(proxy,zg,parsed.qp,context)
            scales=prior['scales'].cpu().numpy()
            coarse=entropy.decode_coarse(parsed.coarse,scales,parsed.width)
            reps=coarse_representatives(coarse,parsed.width)
            bottom=self.bridge.synthesize(proxy,zg,self.tensor(reps),parsed.qp,context)
            brgb=self.rgb(bottom,h,w)
            yrgb=brgb
            if parsed.fine is not None:
                fine=entropy.decode_fine(parsed.fine,coarse,scales,parsed.width)
                symbols=restore_symbols(coarse,fine,parsed.width)
                out=self.bridge.synthesize(proxy,zg,self.tensor(symbols),parsed.qp,context)
                yrgb=self.rgb(out,h,w)
            from tools.latent_probe import context_exact
            if not context_exact(self.bridge,proxy,context): raise RuntimeError('receiver reference changed by display E')
            iframe=rgb_from_tensor(i,h,w)[None]
            return np.concatenate([iframe,brgb]),np.concatenate([iframe,yrgb]),dict(
                source_frames_read=False,frame_count=9,base_reference_unchanged=True,
                base_bytes=parsed.base_end,enhancement_bytes=len(data)-parsed.base_end-parsed.ignored_tail_bytes,
                ignored_tail_bytes=parsed.ignored_tail_bytes,base_header_bytes=wire.HEADER.size+wire.IDENTITIES+wire.BASE_DIGEST,
                I_bytes=len(parsed.intra),z_bytes=len(parsed.z),coarse_y_bytes=len(parsed.coarse),
                refinement_payload_bytes=len(parsed.fine or b''),
                enhancement_header_bytes=wire.EHEADER.size if parsed.fine is not None else 0,
                actual_bytes=len(data),bpp=len(data)*8/(9*h*w),qp_star=parsed.qp,width=parsed.width,
                G_executed=False,G_assets_loaded=False,scope='one I QP32 + P8; full-view refinement')

    def encode(self,source,qp,widths):
        import torch
        from demo.stage_c_three_path_roi_probe import tensor_from_rgb,rgb_from_tensor
        from src.utils.stream_helper import write_sps,write_ip
        if source.dtype!=np.uint8 or len(source)!=9: raise ValueError('nine uint8 RGB frames required')
        h,w=source.shape[1:3]
        pad_r,pad_b=self.codec.i_net.get_padding_size(h,w,16)
        torch.cuda.set_stream(self.codec.stream)
        with torch.inference_mode():
            tensors=[tensor_from_rgb(x,self.codec.device) for x in source]
            i=self.codec.i_net.compress(tensors[0],32,pad_b,pad_r)
            iframe=i['x_hat'].clone()
            self.codec.p_net.add_ref_feature_from_frame(iframe)
            proxy=self.codec.p_net.proxy;context=self.bridge.context(proxy)
            p=self.codec.p_net.compress(torch.cat(tensors[1:],1),qp,False,pad_b,pad_r)
            encoded=self.bridge.encoded(proxy)
            symbols=encoded['symbols'].cpu().numpy().astype(np.int16)
            z=encoded['z'].cpu().numpy()
            prior=self.bridge.prior(proxy,encoded['z'],qp,context)
            scales=prior['scales'].cpu().numpy()
            zdata=entropy.encode_z(z,self.codec.p_net.bit_estimator_z.get_cdf_info(),qp)
            if not np.array_equal(entropy.decode_z(zdata,z.shape,self.codec.p_net.bit_estimator_z.get_cdf_info(),qp),z):
                raise RuntimeError('z rANS roundtrip failed')
            complete=self.bridge.synthesize(proxy,encoded['z'],encoded['symbols'],qp,context)
            native_rgb=np.concatenate([rgb_from_tensor(iframe,h,w)[None],self.rgb(complete,h,w)])
            # Native unlayered comparator, identical bootstrap/reference/workpoint.
            native=io.BytesIO()
            write_sps(native,dict(sps_id=0,height=h,width=w))
            write_ip(native,True,0,32,int(i['ec_parallel']),0,bytes(i['bit_stream']))
            write_ip(native,False,0,qp,int(p['ec_parallel']),0,bytes(p['bit_stream']))
            outputs={}
            for width in widths:
                c,r=split_symbols(symbols,width)
                bdata=entropy.encode_coarse(c,scales,width)
                edata=entropy.encode_fine(c,r,scales,width)
                if not np.array_equal(entropy.decode_coarse(bdata,scales,width),c):
                    raise RuntimeError('coarse rANS roundtrip failed')
                if not np.array_equal(restore_symbols(c,entropy.decode_fine(edata,c,scales,width),width),symbols):
                    raise RuntimeError('refinement rANS roundtrip failed')
                base=wire.base_stream(qp=qp,width=width,height=h,image_width=w,
                    i_parallel=int(i['ec_parallel']),identities=self.identities,
                    intra=bytes(i['bit_stream']),z=zdata,coarse=bdata)
                packet=wire.enhancement_packet(edata)
                outputs[width]=(base,base+packet)
            # This independent native decode also checks the bridge against native
            # packet synthesis for the actual freshly encoded endpoint.
            decoded=self.codec.decode(native.getvalue(),9)
            if not np.array_equal(decoded,native_rgb): raise RuntimeError('native endpoint pixel mismatch')
        return outputs,native.getvalue(),native_rgb
