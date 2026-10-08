// Width-3 refinement rANS with full-width table indexes and checked reads.
// The arithmetic/order follows the MIT-licensed UF rANS in src/cpp/py_rans.
// Separate sidecar: does not rebuild or replace either installed UF extension.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
namespace py = pybind11;
template<class T> using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;
constexpr uint32_t LOWER = 1u << 23;

void validate(const Array<int32_t>& ids, const Array<int32_t>& cdf) {
    if (ids.ndim()!=1 || ids.size()==0 || ids.size()>(1<<26) ||
        cdf.ndim()!=2 || cdf.shape(1)!=5 || cdf.shape(0)<1)
        throw std::invalid_argument("invalid rANS array shapes");
    auto t=cdf.unchecked<2>();
    for (py::ssize_t j=0;j<t.shape(0);++j) {
        if (t(j,0)!=0 || t(j,4)!=65536) throw std::invalid_argument("invalid CDF endpoint");
        for(int k=1;k<5;++k) if(t(j,k)<=t(j,k-1))
            throw std::invalid_argument("non-increasing CDF");
    }
    for(auto i=0;i<ids.size();++i) if(ids.data()[i]<0 || ids.data()[i]>=cdf.shape(0))
        throw std::invalid_argument("CDF index out of bounds");
}

py::bytes encode(const Array<int16_t>& values, const Array<int32_t>& ids,
                 const Array<int32_t>& cdf) {
    validate(ids,cdf);
    if(values.ndim()!=1 || values.size()!=ids.size()) throw std::invalid_argument("symbol shape");
    uint32_t state=LOWER;
    std::vector<uint8_t> emitted;
    emitted.reserve(values.size()*2);
    auto t=cdf.unchecked<2>();
    for(py::ssize_t i=values.size();i-- >0;) {
        int v=values.data()[i];
        if(v < -1 || v > 1) throw std::invalid_argument("width3 symbol outside alphabet");
        int rank=std::abs(v)*2-(v>0), row=ids.data()[i];
        uint32_t start=t(row,rank), freq=t(row,rank+1)-start;
        while(state >= (freq<<15)) { emitted.push_back(state&255); state>>=8; }
        state=((state/freq)<<16)+(state%freq)+start;
    }
    std::string bytes;
    bytes.reserve(4+emitted.size());
    for(int j=0;j<4;++j) bytes.push_back((state>>(8*j))&255);
    for(auto i=emitted.rbegin();i!=emitted.rend();++i) bytes.push_back(*i);
    return py::bytes(bytes);
}

Array<int16_t> decode(const py::bytes& payload, const Array<int32_t>& ids,
                      const Array<int32_t>& cdf) {
    validate(ids,cdf);
    std::string bytes=payload;
    if(bytes.size()<4) throw std::invalid_argument("short rANS state");
    uint32_t state=0;
    for(int j=0;j<4;++j) state|=uint32_t(uint8_t(bytes[j]))<<(8*j);
    if(state<LOWER || state>=(1u<<31)) throw std::invalid_argument("invalid rANS state");
    size_t cursor=4;
    Array<int16_t> result(ids.size()); auto t=cdf.unchecked<2>();
    for(py::ssize_t i=0;i<ids.size();++i) {
        uint32_t q=state&65535, row=ids.data()[i];
        int rank=0;
        while(rank<3 && uint32_t(t(row,rank+1))<=q) ++rank;
        if(rank==3) throw std::invalid_argument("unexpected escape in width3 stream");
        uint32_t start=t(row,rank),freq=t(row,rank+1)-start;
        state=freq*(state>>16)+q-start;
        while(state<LOWER) {
            if(cursor==bytes.size()) throw std::invalid_argument("truncated rANS payload");
            state=(state<<8)|uint8_t(bytes[cursor++]);
        }
        result.mutable_data()[i]=rank==0 ? 0 : (rank==1 ? 1 : -1);
    }
    if(cursor!=bytes.size() || state!=LOWER) throw std::invalid_argument("noncanonical rANS termination");
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("encode",&encode); m.def("decode",&decode); }
