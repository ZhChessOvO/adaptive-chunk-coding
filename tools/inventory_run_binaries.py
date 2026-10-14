"""Read-only size inventory to scope migration-oriented run cache retirement."""
import json
import os
from collections import Counter
from pathlib import Path
from tools.retire_run_caches import RUNS, write


def main():
    names, roots, examples = Counter(), Counter(), {}
    for parent, dirs, files in os.walk(RUNS, followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(parent)/n).is_symlink()]
        for name in files:
            p=Path(parent)/name
            if p.suffix not in ('.npz','.npy','.pt','.acse','.acsg','.bin') or p.is_symlink():continue
            size=p.stat().st_size
            names[name]+=size; roots[p.relative_to(RUNS).parts[0]]+=size
            examples.setdefault(name,[])
            if len(examples[name])<3:examples[name].append(str(p))
    result=dict(names=[dict(name=n,bytes=v,examples=examples[n]) for n,v in names.most_common()],
                roots=dict(roots.most_common()))
    write(RUNS/'_maintenance_20261015/binary_inventory.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
