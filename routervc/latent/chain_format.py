"""RVLC1 diagnostic multi-chunk B followed by optional chunk E packets.

The compact single-P8 RVL1 remains unchanged. This separate research format
charges its JSON manifest and checksums; it is not an overhead-optimized format.
All decoded region coverage is implicit in packet IDs, not another mask.
"""
import hashlib
import json
import math
import struct

MAGIC=b'RVLCHAIN'
HEADER=struct.Struct('<8sI')
EHEADER=struct.Struct('<4sII32s')


def hash_bytes(data): return hashlib.sha256(data).digest()


def pack(meta,intra,chunks):
    meta=dict(meta,I_size=len(intra),chunks=[dict(z_size=len(z),y_size=len(y)) for z,y in chunks])
    text=json.dumps(meta,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    data=HEADER.pack(MAGIC,len(text))+text+intra+b''.join(z+y for z,y in chunks)
    result=data+hash_bytes(data)
    parse(result)
    return result


def packet(index,fine):
    if type(index) is not int or index < 0 or len(fine)<8: raise ValueError('invalid E packet')
    return EHEADER.pack(b'CE01',index,len(fine),hash_bytes(fine))+fine


def parse(data,allow_incomplete_tail=False):
    if len(data)<HEADER.size: raise ValueError('short chain header')
    magic,size=HEADER.unpack_from(data)
    if magic!=MAGIC or size < 10 or size > 65536 or len(data)<HEADER.size+size: raise ValueError('invalid chain header')
    meta=json.loads(data[HEADER.size:HEADER.size+size])
    expected={'profile','identities','frames','height','width','q_star','bin_width','I_qp','I_parallel','I_size','chunks'}
    if set(meta)!=expected or meta['profile']!='RVLC1' or meta['q_star']!=48 or meta['bin_width']!=3 or meta['I_qp']!=32:
        raise ValueError('unsupported chain profile')
    if any(type(meta[k]) is not int for k in ('frames','height','width','q_star','bin_width','I_qp','I_parallel','I_size')):
        raise ValueError('noninteger chain config')
    if not 2<=meta['frames']<=65 or any(v<64 or v>2048 or v%16 for v in (meta['height'],meta['width'])):
        raise ValueError('unsupported chain geometry')
    if not 1<=meta['I_parallel']<=8 or not 4<=meta['I_size']<=64<<20: raise ValueError('invalid bootstrap')
    if (len(meta['identities'])!=5 or any(not isinstance(h,str) or len(h)!=64 for h in meta['identities'])):
        raise ValueError('invalid identities')
    for identity in meta['identities']: bytes.fromhex(identity)
    if len(meta['chunks'])!=math.ceil((meta['frames']-1)/8): raise ValueError('wrong P8 count')
    cursor=HEADER.size+size;start=cursor;cursor+=meta['I_size']
    chunks=[]
    for chunk in meta['chunks']:
        if set(chunk)!= {'z_size','y_size'} or any(type(v) is not int or not 4<=v<=64<<20 for v in chunk.values()):
            raise ValueError('invalid chunk lengths')
        z_end=cursor+chunk['z_size'];y_end=z_end+chunk['y_size']
        chunks.append((data[cursor:z_end],data[z_end:y_end]));cursor=y_end
    end=cursor+32
    if end>len(data) or data[cursor:end]!=hash_bytes(data[:cursor]): raise ValueError('incomplete/corrupt B')
    intra=data[start:start+meta['I_size']];cursor=end;enhancements={};ignored=0;duplicates=0
    while cursor<len(data):
        remain=len(data)-cursor
        if remain<EHEADER.size:
            if not allow_incomplete_tail: raise ValueError('incomplete E header')
            ignored=remain;break
        tag,index,length,digest=EHEADER.unpack_from(data,cursor)
        if tag!=b'CE01' or index>=len(chunks) or not 8<=length<=64<<20: raise ValueError('invalid E packet header')
        if remain<EHEADER.size+length:
            if not allow_incomplete_tail: raise ValueError('incomplete E body')
            ignored=remain;break
        payload=data[cursor+EHEADER.size:cursor+EHEADER.size+length]
        if hash_bytes(payload)!=digest: raise ValueError('corrupt E')
        if index in enhancements:
            if enhancements[index]!=payload: raise ValueError('conflicting repeated E')
            duplicates+=1
        enhancements[index]=payload;cursor+=EHEADER.size+length
    return dict(meta=meta,intra=intra,chunks=chunks,enhancements=enhancements,
                base_end=end,ignored_tail_bytes=ignored,duplicates=duplicates)
