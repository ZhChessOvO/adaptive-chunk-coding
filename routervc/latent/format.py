"""RVL1: one charged I bootstrap + coarse P8, followed by one optional E packet.

Research profile only. Spatial E, multiple chunks and G are future extensions,
NOT silently approximated by this parser. Byte truncation is allowed only at a
complete B or E boundary; an incomplete E may explicitly fall back to B.
"""
from dataclasses import dataclass
import hashlib
import struct

MAGIC=b'RVLAT001'
# magic, version, I/P quality, width, H/W, I parallel, reserved, I/z/y lengths
HEADER=struct.Struct('<8s4BHHBBIII')
EHEADER=struct.Struct('<4sI32s')
IDENTITIES=128  # image, video, native library, profile source SHA256
BASE_DIGEST=32


def digest(data): return hashlib.sha256(data).digest()


@dataclass(frozen=True)
class Parsed:
    qp: int
    width: int
    height: int
    image_width: int
    i_parallel: int
    identities: tuple[str,...]
    intra: bytes
    z: bytes
    coarse: bytes
    fine: bytes | None
    base_end: int
    ignored_tail_bytes: int


def base_stream(*,qp,width,height,image_width,i_parallel,identities,intra,z,coarse):
    if len(identities)!=4: raise ValueError('four shared identities required')
    header=HEADER.pack(MAGIC,1,32,qp,width,height,image_width,i_parallel,0,
                       len(intra),len(z),len(coarse))
    ids=b''.join(bytes.fromhex(h) for h in identities)
    if len(ids)!=IDENTITIES: raise ValueError('invalid identity digests')
    data=header+ids+intra+z+coarse
    wire=data+digest(data)
    parse(wire)
    return wire


def enhancement_packet(fine):
    if len(fine)<8: raise ValueError('empty refinement')
    return EHEADER.pack(b'ENH1',len(fine),digest(fine))+fine


def parse(data,*,allow_incomplete_tail=False):
    if len(data)<HEADER.size+IDENTITIES+BASE_DIGEST: raise ValueError('truncated base header')
    magic,version,iqp,qp,width,h,w,parallel,reserved,ni,nz,ny=HEADER.unpack_from(data)
    if magic!=MAGIC or version!=1 or iqp!=32 or reserved!=0: raise ValueError('unknown latent profile')
    if qp not in (32,48) or width not in (3,9,17): raise ValueError('unsupported diagnostic q/width')
    if any(x < 64 or x > 2048 or x%16 for x in (h,w)): raise ValueError('unsupported geometry')
    if not 1<=parallel<=8 or min(ni,nz,ny)<4: raise ValueError('invalid entropy framing')
    cursor=HEADER.size
    ids=tuple(data[cursor+i*32:cursor+(i+1)*32].hex() for i in range(4))
    cursor+=IDENTITIES
    end=cursor+ni+nz+ny+BASE_DIGEST
    if end>len(data) or max(ni,nz,ny)>64<<20: raise ValueError('truncated/oversized base payload')
    if data[end-BASE_DIGEST:end]!=digest(data[:end-BASE_DIGEST]): raise ValueError('base checksum mismatch')
    intra=data[cursor:cursor+ni];cursor+=ni
    z=data[cursor:cursor+nz];cursor+=nz
    coarse=data[cursor:cursor+ny]
    fine=None;ignored=0
    tail=data[end:]
    if tail:
        if len(tail)<EHEADER.size:
            if not allow_incomplete_tail: raise ValueError('truncated E header')
            ignored=len(tail)
        else:
            emagic,size,sha=EHEADER.unpack_from(tail)
            if emagic!=b'ENH1' or size < 8 or size > 64<<20: raise ValueError('invalid E header')
            if len(tail) < EHEADER.size+size:
                if not allow_incomplete_tail: raise ValueError('truncated E payload')
                ignored=len(tail)
            elif len(tail)!=EHEADER.size+size: raise ValueError('extra/duplicate E packets')
            else:
                fine=tail[EHEADER.size:]
                if digest(fine)!=sha: raise ValueError('E checksum mismatch')
    return Parsed(qp,width,h,w,parallel,ids,intra,z,coarse,fine,end,ignored)
