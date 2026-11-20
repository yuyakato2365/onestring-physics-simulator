"""Exercise the same native cut guard used by the OptCuts fork."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_cut_connectivity(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'connectivity.cpp'
    source.write_text(r'''
#include <array>
#include <cassert>
#include "OneStringSeamConnectivity.hpp"
struct Faces {
    std::vector<std::array<int,3>> f;
    int rows() const { return int(f.size()); }
    int operator()(int i, int k) const { return f[i][k]; }
};
std::map<std::pair<int,int>,int> edges(const Faces& f) {
    std::map<std::pair<int,int>,int> e;
    for (int i=0;i<f.rows();++i)
        for (int k=0;k<3;++k) e[{f(i,k),f(i,(k+1)%3)}]=i;
    return e;
}
int main() {
    // Cutting the diagonal of a disk separates its two triangles.
    Faces disk{{{{0,1,2}},{{0,2,3}}}};
    assert(onestring::cutDisconnects(disk,edges(disk),0,2));
    assert(onestring::cutDisconnects(disk,edges(disk),2,0));
    // Starting/growing a slit to an interior vertex remains connected.
    Faces fan{{{{0,1,4}},{{1,2,4}},{{2,3,4}},{{3,0,4}}}};
    assert(!onestring::cutDisconnects(fan,edges(fan),0,4));
    // Connecting outer and inner boundaries opens an annulus into ONE disk.
    Faces annulus{{{{0,1,5}},{{0,5,4}},{{1,2,6}},{{1,6,5}},
                    {{2,3,7}},{{2,7,6}},{{3,0,4}},{{3,4,7}}}};
    assert(!onestring::cutDisconnects(annulus,edges(annulus),0,4));
    // Existing boundary edges have no two incident faces to disconnect.
    assert(!onestring::cutDisconnects(disk,edges(disk),0,1));
}
''')
    exe = tmp_path / 'connectivity'
    subprocess.run([compiler,'-std=c++11','-I',str(root/'third_party/OptCuts_csf_onestring/src'),
                    str(source),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)


def test_single_chart_mode_enables_and_clears_native_guard(monkeypatch):
    from onestring_physics import optcuts_soft_fit_20261104 as sf
    import os
    monkeypatch.setattr(sf, '_STATE', dict(sf._STATE))
    monkeypatch.setattr(sf, '_install', lambda: None)
    for key in (*sf._ALL_KEYS, sf._MARK):
        monkeypatch.delenv(key, raising=False)
    sf.configure(True, constraint='soft', multistart=True)
    assert os.environ['ONESTRING_KEEP_CONNECTED'] == '1'
    assert os.environ['ONESTRING_ANGLE_W'] == '16'
    sf.configure(False)
    assert 'ONESTRING_KEEP_CONNECTED' not in os.environ
