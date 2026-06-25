#!/usr/bin/env python3
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--name',required=True); p.add_argument('--repo',required=True); p.add_argument('--output',required=True); a=p.parse_args()
repo=Path(a.repo).resolve(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
def git(*args):
    try:return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,timeout=20).stdout.strip()
    except Exception:return ''
files=[]
if repo.exists():
    for f in sorted(repo.rglob('*')):
        if f.is_file() and '.git' not in f.parts:
            files.append(str(f.relative_to(repo)))
record={'name':a.name,'repo':str(repo),'exists':repo.exists(),'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain')),'epoch':time.time(),'files':files[:2000],'n_files':len(files),'readmes':[str(x.relative_to(repo)) for x in repo.glob('README*')] if repo.exists() else []}
out.write_text(json.dumps(record,indent=2)); print(out)
