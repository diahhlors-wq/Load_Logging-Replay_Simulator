
#!/usr/bin/env python3
import argparse, csv, math, sys, json
from collections import defaultdict

def mask_width(v, w):
    if w == 1: return v & 0xFF
    if w == 2: return v & 0xFFFF
    if w == 4: return v & 0xFFFFFFFF
    return v & 0xFFFFFFFFFFFFFFFF

class SetAssocTable:
    def __init__(self, entries=1024, assoc=4, idx_shift=2):
        assert entries % assoc == 0 and entries > 0 and assoc > 0
        self.entries = entries
        self.assoc   = assoc
        self.sets    = entries // assoc
        # require power-of-two sets for mask; otherwise use modulo
        if (self.sets & (self.sets - 1)) == 0:
            self.set_mask = self.sets - 1
            self.power_of_two = True
        else:
            self.set_mask = None
            self.power_of_two = False
        self.idx_shift = idx_shift
        self.tag_shift = idx_shift + (self.sets - 1).bit_length() if self.power_of_two else idx_shift
        self.rr = [0]*self.sets if self.sets > 0 else [0]
        # storage
        self.tab = [None]*entries  # store dicts with keys we need

    def _set_index(self, pc):
        if self.power_of_two:
            return (pc >> self.idx_shift) & self.set_mask
        else:
            return ((pc >> self.idx_shift) % self.sets) if self.sets else 0

    def _base(self, s): return s * self.assoc

    def find(self, pc):
        if self.sets == 0: return -1, -1
        s = self._set_index(pc)
        base = self._base(s)
        tag = pc >> self.tag_shift
        for w in range(self.assoc):
            e = self.tab[base+w]
            if e is not None and e['valid'] and e['tag'] == tag:
                return s, base+w
        return s, -1

    def victim(self, s):
        base = self._base(s)
        v = self.rr[s]
        self.rr[s] = (self.rr[s] + 1) % self.assoc
        return base + v

def simulate(csv_path, entries=1024, assoc=4, conf_thresh=2, chooser_bias=1, eligible_field='eligible'):
    # Tables
    lv = SetAssocTable(entries, assoc)
    ls = SetAssocTable(entries, assoc)

    # Stats
    total_loads = 0
    eligible_loads = 0
    preds = 0
    confident = 0
    used = 0
    correct = 0
    used_lv = correct_lv = 0
    used_ls = correct_ls = 0

    # Per-PC (optional insights)
    per_pc = defaultdict(lambda: {'total':0,'used':0,'correct':0})

    with open(csv_path, 'r', newline='') as f:
        rdr = csv.DictReader(f)
        # basic header sanity
        required = {'pc','width','value'}
        if not required.issubset(rdr.fieldnames):
            raise SystemExit(f"CSV must have columns: {required}. Got: {rdr.fieldnames}")
        for row in rdr:
            try:
                pc_raw = row['pc']
                pc = int(pc_raw, 16) if isinstance(pc_raw, str) and pc_raw.lower().startswith(('0x','-0x')) else int(pc_raw)
                width = int(row['width'])
                val = int(row['value'])
            except Exception as e:
                print(f"Skipping malformed row: {row} ({e})", file=sys.stderr)
                continue

            total_loads += 1
            per_pc[pc]['total'] += 1

            eligible = 1
            if eligible_field in row:
                try:
                    eligible = int(row[eligible_field])
                except: eligible = 1
            # Common filters if present
            if 'mmio' in row:
                try:
                    if int(row['mmio']) != 0: eligible = 0
                except: pass
            if 'alias_risk' in row:
                try:
                    if int(row['alias_risk']) != 0: eligible = 0
                except: pass

            if eligible: eligible_loads += 1

            # --- Predict (measure-only)
            # LV candidate
            s_lv, idx_lv = lv.find(pc)
            lv_ok = (idx_lv != -1)
            lv_val = 0; lv_conf = 0
            if lv_ok:
                e = lv.tab[idx_lv]
                lv_val = mask_width(e['last'], width)
                lv_conf = e['conf']

            # LS candidate
            s_ls, idx_ls = ls.find(pc)
            ls_ok = (idx_ls != -1) and (ls.tab[idx_ls]['have_two'])
            ls_val = 0; ls_conf = 0
            if ls_ok:
                e = ls.tab[idx_ls]
                ls_val = mask_width(e['last'] + e['stride'], width)
                ls_conf = e['conf']

            chose = None; pred_val = None; pred_conf = 0
            if lv_ok or ls_ok:
                preds += 1
                # chooser: prefer LS only if conf_ls >= conf_lv + chooser_bias
                if ls_ok and (ls_conf >= lv_conf + chooser_bias):
                    chose = 'ls'; pred_val = ls_val; pred_conf = ls_conf
                elif lv_ok:
                    chose = 'lv'; pred_val = lv_val; pred_conf = lv_conf
                else:
                    chose = 'ls'; pred_val = ls_val; pred_conf = ls_conf

            # Used iff confident and eligible
            used_this = False; correct_this = False
            if chose is not None and pred_conf >= conf_thresh and eligible:
                confident += 1
                used += 1; used_this = True
                if pred_val == mask_width(val, width):
                    correct += 1; correct_this = True
                if chose == 'lv':
                    used_lv += 1; 
                    if correct_this: correct_lv += 1
                else:
                    used_ls += 1; 
                    if correct_this: correct_ls += 1

            if used_this:
                per_pc[pc]['used'] += 1
                if correct_this: per_pc[pc]['correct'] += 1

            # --- Train
            # LV train
            if idx_lv == -1:
                # insert
                v = lv.victim(s_lv if lv.sets else 0)
                lv.tab[v] = {'tag': (pc >> lv.tag_shift), 'last': mask_width(val,width), 'conf': 1, 'valid': True}
            else:
                e = lv.tab[idx_lv]
                if mask_width(e['last'], width) == mask_width(val, width):
                    if e['conf'] < 3: e['conf'] += 1
                else:
                    if e['conf'] > 0: e['conf'] -= 1
                    e['last'] = mask_width(val, width)
                e['valid'] = True

            # LS train
            if idx_ls == -1:
                v = ls.victim(s_ls if ls.sets else 0)
                ls.tab[v] = {'tag': (pc >> ls.tag_shift), 'last': mask_width(val,width), 'stride': 0, 'conf': 0, 'have_two': False, 'valid': True}
            else:
                e = ls.tab[idx_ls]
                if e['valid']:
                    if e['have_two']:
                        new_stride = (mask_width(val,width) - mask_width(e['last'],width)) & 0xFFFFFFFFFFFFFFFF
                        if new_stride == e['stride']:
                            if e['conf'] < 3: e['conf'] += 1
                        else:
                            if e['conf'] > 0: e['conf'] -= 1
                            e['stride'] = new_stride
                        e['last'] = mask_width(val,width)
                    else:
                        e['stride'] = (mask_width(val,width) - mask_width(e['last'],width)) & 0xFFFFFFFFFFFFFFFF
                        e['have_two'] = True
                        e['last'] = mask_width(val,width)
                else:
                    e.update({'tag': (pc >> ls.tag_shift), 'last': mask_width(val,width), 'stride': 0, 'conf': 0, 'have_two': False, 'valid': True})

    out = {
        'total_loads': total_loads,
        'eligible_loads': eligible_loads,
        'preds': preds,
        'confident': confident,
        'used': used,
        'correct': correct,
        'accuracy': (float(correct)/used) if used else 0.0,
        'coverage_overall': (float(used)/total_loads) if total_loads else 0.0,
        'coverage_eligible': (float(used)/eligible_loads) if eligible_loads else 0.0,
        'effective': (float(correct)/total_loads) if total_loads else 0.0,
        'used_lv': used_lv, 'correct_lv': correct_lv,
        'used_ls': used_ls, 'correct_ls': correct_ls,
        'accuracy_lv': (float(correct_lv)/used_lv) if used_lv else 0.0,
        'accuracy_ls': (float(correct_ls)/used_ls) if used_ls else 0.0
    }
    return out, per_pc

def main():
    p = argparse.ArgumentParser(description="Offline LV+LS predictor evaluation from gem5 load log CSV")
    p.add_argument('csv', help='CSV with columns: pc,width,value[,eligible,mmio,alias_risk]')
    p.add_argument('--entries', type=int, default=1024)
    p.add_argument('--assoc', type=int, default=4)
    p.add_argument('--conf', type=int, default=2, help='confidence threshold (0..3)')
    p.add_argument('--chooser-bias', type=int, default=1, help='LS wins if conf_ls >= conf_lv + bias')
    p.add_argument('--eligible-field', default='eligible', help='column name for eligibility')
    p.add_argument('--json', default=None, help='optional path to write JSON summary')
    p.add_argument('--top-pc', type=int, default=0, help='print top-N PCs by used predictions')
    args = p.parse_args()

    out, per_pc = simulate(args.csv, args.entries, args.assoc, args.conf, args.chooser_bias, args.eligible_field)

    print("== LV+LS Offline Evaluation ==")
    for k in ['total_loads','eligible_loads','preds','confident','used','correct']:
        print(f"{k:>18}: {out[k]}")
    for k in ['accuracy','coverage_overall','coverage_eligible','effective','accuracy_lv','accuracy_ls']:
        print(f"{k:>18}: {out[k]:.4f}")
    print(f"{'used_lv':>18}: {out['used_lv']}   {'used_ls':>10}: {out['used_ls']}")

    if args.top_pc > 0:
        ranked = sorted(per_pc.items(), key=lambda kv: kv[1]['used'], reverse=True)[:args.top_pc]
        print("\nTop PCs by used predictions:")
        for pc, d in ranked:
            acc = (d['correct']/d['used']) if d['used'] else 0.0
            print(f"PC=0x{pc:x}  total={d['total']} used={d['used']} correct={d['correct']} acc={acc:.3f}")

    if args.json:
        with open(args.json, 'w') as jf:
            json.dump(out, jf, indent=2)

if __name__ == '__main__':
    main()
