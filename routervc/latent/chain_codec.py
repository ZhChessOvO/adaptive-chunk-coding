"""Native two-level latent coding with ONLY B in the temporal reference loop."""
from pathlib import Path
import hashlib

import numpy as np
from routervc.latent.codec import LatentCodec,profile_hash
from routervc.latent.transition import load_transition
from routervc.latent import chain_format as fmt,entropy
from routervc.latent.split import split_symbols,restore_symbols,coarse_representatives
from demo.scalable_codec import file_hash


def chain_hash():
    files=('chain_codec.py','chain_format.py','transition.py','transition_bridge.cpp')
    return hashlib.sha256((profile_hash()+''.join(file_hash(Path(__file__).with_name(n)) for n in files)).encode()).hexdigest()


def reference_hash(context):
    # Only initialized, semantically live reference tensors; no dead workspaces.
    result={}
    for name in ('memory','ctx'):
        result[name]=hashlib.sha256(context[name].contiguous().cpu().numpy().tobytes()).hexdigest()
    return result


class ChainCodec(LatentCodec):
    def __init__(self):
        super().__init__()
        self.transition=load_transition()
        self.chain_identities=(*self.identities,chain_hash())

    def advance(self,context,feature,reset):
        proxy=self.codec.p_net.proxy
        self.bridge.restore(proxy,context)
        self.transition.supply_base(proxy,feature,reset)
        decoder_state=self.bridge.context(proxy)
        self.transition.prepare_encoder(proxy)
        ready=self.bridge.context(proxy)
        return ready,decoder_state

    def decode_chain(self,data,*,allow_incomplete_tail=False):
        import torch
        from demo.stage_c_three_path_roi_probe import rgb_from_tensor
        from tools.latent_probe import context_exact
        parsed=fmt.parse(data,allow_incomplete_tail);m=parsed['meta']
        if tuple(m['identities'])!=self.chain_identities: raise ValueError('chain identity mismatch')
        h,w,count=m['height'],m['width'],m['frames'];base=[];shown=[];refs=[]
        torch.cuda.set_stream(self.codec.stream)
        with torch.inference_mode():
            intra=self.codec.i_net.decompress(parsed['intra'],{'height':h,'width':w},32,m['I_parallel'])['x_hat'].clone()
            self.codec.p_net.add_ref_feature_from_frame(intra)
            proxy=self.codec.p_net.proxy;ready=self.bridge.context(proxy)
            frame=rgb_from_tensor(intra,h,w);base.append(frame);shown.append(frame)
            for index,(zdata,ydata) in enumerate(parsed['chunks']):
                start=1+index*8;valid=min(8,count-start);reset=(start+8)%32==1
                z=entropy.decode_z(zdata,(1,128,(h+63)//64,(w+63)//64),self.codec.p_net.bit_estimator_z.get_cdf_info(),48)
                zg=self.tensor(z);scales=self.bridge.prior(proxy,zg,48,ready)['scales'].cpu().numpy()
                c=entropy.decode_coarse(ydata,scales,3)
                bottom=self.bridge.synthesize(proxy,zg,self.tensor(coarse_representatives(c,3)),48,ready)
                b=self.rgb(bottom,h,w)[:valid];base.extend(b)
                display=b
                if index in parsed['enhancements']:
                    r=entropy.decode_fine(parsed['enhancements'][index],c,scales,3)
                    full=self.bridge.synthesize(proxy,zg,self.tensor(restore_symbols(c,r,3)),48,ready)
                    display=self.rgb(full,h,w)[:valid]
                if not context_exact(self.bridge,proxy,ready): raise RuntimeError('E mutated B reference before commit')
                shown.extend(display)
                ready,_=self.advance(ready,bottom['feature'],reset)
                refs.append(reference_hash(ready))
            torch.cuda.synchronize()
        return np.stack(base),np.stack(shown),dict(base_reference_hashes=refs,
            base_reference_unchanged=True,source_frames_read=False,frame_count=count,
            received_chunks=sorted(parsed['enhancements']),duplicates=parsed['duplicates'],
            ignored_tail_bytes=parsed['ignored_tail_bytes'],base_bytes=parsed['base_end'],
            actual_bytes=len(data),bpp=len(data)*8/(count*h*w),generation=False,router=False,
            header_and_checksum_bytes=parsed['base_end']-len(parsed['intra'])-sum(len(z)+len(y) for z,y in parsed['chunks']),
            I_bytes=len(parsed['intra']),z_bytes=sum(len(z) for z,_ in parsed['chunks']),
            coarse_y_bytes=sum(len(y) for _,y in parsed['chunks']),
            E_wire_bytes=len(data)-parsed['base_end']-parsed['ignored_tail_bytes'])

    def encode_chain(self,source):
        import torch
        from demo.stage_c_three_path_roi_probe import tensor_from_rgb,rgb_from_tensor,encode_dcvc_stream
        if source.dtype!=np.uint8 or not 2<=len(source)<=65: raise ValueError('invalid source frames')
        h,w=source.shape[1:3];count=len(source);pad_r,pad_b=self.codec.i_net.get_padding_size(h,w,16)
        torch.cuda.set_stream(self.codec.stream)
        with torch.inference_mode():
            i=self.codec.i_net.compress(tensor_from_rgb(source[0],self.codec.device),32,pad_b,pad_r)
            iframe=i['x_hat'].clone();proxy=self.codec.p_net.proxy
            self.codec.p_net.add_ref_feature_from_frame(iframe,apply_feature_adaptor=False)
            proxy=self.codec.p_net.proxy;decoder_state=self.bridge.context(proxy)
            self.codec.p_net.add_ref_feature_from_frame(iframe)
            ready=self.bridge.context(proxy)
            first=rgb_from_tensor(iframe,h,w);base=[first];full_frames=[first]
            chunks=[];packets=[];refs=[];same_context_native=[]
            for index,start in enumerate(range(1,count,8)):
                valid=min(8,count-start);reset=(start+8)%32==1
                frames=[tensor_from_rgb(s,self.codec.device) for s in source[start:start+valid]]
                frames+= [frames[-1]]*(8-valid)
                self.bridge.restore(proxy,ready)
                p=self.codec.p_net.compress(torch.cat(frames,1),48,reset,pad_b,pad_r)
                enc=self.bridge.encoded(proxy)
                y=enc['symbols'].cpu().numpy().astype(np.int16);z=enc['z'].cpu().numpy()
                scales=self.bridge.prior(proxy,enc['z'],48,ready)['scales'].cpu().numpy()
                full=self.bridge.synthesize(proxy,enc['z'],enc['symbols'],48,ready)
                full_rgb=self.rgb(full,h,w)[:valid]
                # Independently run the unmodified native P decoder starting from
                # its pre-adaptation B-only DPB. It must agree chunk by chunk.
                self.bridge.restore(proxy,decoder_state)
                native=self.codec.p_net.decompress(p['bit_stream'],{'height':h,'width':w},48,p['ec_parallel'],reset)['x_hat']
                check=np.stack([rgb_from_tensor(x,h,w) for x in native[:valid]])
                if not np.array_equal(check,full_rgb): raise RuntimeError('continuous same-context native endpoint mismatch')
                self.bridge.restore(proxy,ready)
                c,r=split_symbols(y,3)
                bottom=self.bridge.synthesize(proxy,enc['z'],self.tensor(coarse_representatives(c,3)),48,ready)
                base.extend(self.rgb(bottom,h,w)[:valid]);full_frames.extend(full_rgb)
                zdata=entropy.encode_z(z,self.codec.p_net.bit_estimator_z.get_cdf_info(),48)
                ydata=entropy.encode_coarse(c,scales,3);edata=entropy.encode_fine(c,r,scales,3)
                chunks.append((zdata,ydata));packets.append(fmt.packet(index,edata))
                same_context_native.append(dict(native_P_bytes=len(p['bit_stream']),valid_frames=valid,
                                                endpoint_exact=True,reset=reset))
                ready,decoder_state=self.advance(ready,bottom['feature'],reset)
                refs.append(reference_hash(ready))
            meta=dict(profile='RVLC1',identities=list(self.chain_identities),frames=count,height=h,width=w,
                      q_star=48,bin_width=3,I_qp=32,I_parallel=int(i['ec_parallel']))
            b=fmt.pack(meta,bytes(i['bit_stream']),chunks)
            # Actual conventional native-UF own-reference chain; not the endpoint
            # anchor above. Its better full references can change rate AND pixels.
            native,_=encode_dcvc_stream(list(source),32,48,self.codec.i_net,self.codec.p_net,self.codec.device,32)
            own=self.codec.decode(native,count)
        return b,packets,np.stack(base),np.stack(full_frames),native,own,dict(
            base_reference_hashes=refs,same_context_chunks=same_context_native)
