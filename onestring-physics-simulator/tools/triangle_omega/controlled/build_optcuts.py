"""Build the supplied OptCuts fork alongside the unchanged Mac binary."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import shlex
import sys

here = Path(__file__).resolve().parent
repo = here.parents[2]
source = repo/'third_party/OptCuts_csf_onestring'
build = source/'build2'
dest = Path(sys.argv[1]).resolve()
dest.mkdir(parents=True, exist_ok=True)
archive = subprocess.check_output(['git','-C',str(source),'archive','cd23026','src','CMakeLists.txt'])
subprocess.run(['tar','xf','-','-C',str(dest)],input=archive,check=True)
for patch in ('optcuts_fork_onestring.patch','optcuts_convergence.patch','optcuts_disk_checkpoint.patch'):
    subprocess.run(['patch','-p1','-i',str(here/patch)],cwd=dest,check=True)
config = build/'CMakeFiles/OptCuts_bin.dir'
flags = dict(line.split(' = ',1) for line in (config/'flags.make').read_text().splitlines() if ' = ' in line)
includes = flags['CXX_INCLUDES'].replace(str(source/'src'), str(dest/'src'))
command = shlex.split((config/'link.txt').read_text())
def compile_one(name):
    obj = dest/(name+'.o')
    file = dest/'src'/name
    subprocess.run(['/usr/bin/c++',*shlex.split(flags['CXX_FLAGS']),*shlex.split(includes),'-c',str(file),'-o',str(obj)],check=True)
    return str(obj)
(dest/'Energy').mkdir(exist_ok=True)
files=['main.cpp','TriMesh.cpp','Optimizer.cpp','Energy/SymDirichletEnergy.cpp','Energy/NeckEnergy.cpp']
with ThreadPoolExecutor(max_workers=4) as pool:
    objs=list(pool.map(compile_one,files))
for file,obj in zip(files[:-1],objs[:-1]):
    command[command.index('CMakeFiles/OptCuts_bin.dir/src/'+file+'.o')]=obj
command.insert(command.index('-o'),objs[-1])
command[command.index('-o')+1]=str(build/'OptCuts_neck_bin')
subprocess.run(command,cwd=build,check=True)
