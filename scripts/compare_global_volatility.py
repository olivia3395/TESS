"""对比两个数据集版本中test.json的global_volatility字段差异。"""
import json
from pathlib import Path
import sys

def main():
    base = Path(__file__).resolve().parents[1] / 'dataset' / 'FNSPID'
    camf_path = base / 'ver_camf_global_volatility' / 'test.json'
    sync_path = base / 'label' / 'ver_synchronized_volatility' / 'test.json'
    
    print(f"Reading {camf_path}...")
    sys.stdout.flush()
    with camf_path.open('r', encoding='utf-8') as f:
        camf = json.load(f)
    
    print(f"Reading {sync_path}...")
    sys.stdout.flush()
    with sync_path.open('r', encoding='utf-8') as f:
        sync = json.load(f)
    
    if len(camf) != len(sync):
        print(f'WARNING: len mismatch, camf={len(camf)}, sync={len(sync)}')
    
    n = min(len(camf), len(sync))
    print(f"Comparing {n} samples...")
    sys.stdout.flush()
    
    diff_indices = []
    for i in range(n):
        v1 = camf[i].get('global_volatility')
        v2 = sync[i].get('global_volatility')
        if v1 != v2:
            diff_indices.append({
                'index': i, 
                'camf_global_volatility': v1, 
                'sync_global_volatility': v2
            })
    
    out_dir = base / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'diff_indices_global_volatility_camf_vs_sync_test.json'
    
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(diff_indices, f, ensure_ascii=False, indent=2)
    
    print(f'\nResults:')
    print(f'  Total samples compared: {n}')
    print(f'  Different global_volatility count: {len(diff_indices)}')
    print(f'  Result saved to: {out_path}')
    sys.stdout.flush()
    
    if len(diff_indices) > 0:
        print(f'\nFirst 10 differences:')
        for item in diff_indices[:10]:
            print(f"  Index {item['index']}: {item['camf_global_volatility']} -> {item['sync_global_volatility']}")
        sys.stdout.flush()

if __name__ == '__main__':
    main()

