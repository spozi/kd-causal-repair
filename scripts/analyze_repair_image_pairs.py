"""Report every fixed CE-KD/repaired-KD pair from saved correctness counts."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from scripts.repair_significance import SOURCE, OUT, ROOT, holm
from scipy.stats import binomtest


def main():
    rows = []
    with tarfile.open(SOURCE) as archive:
        snapshot = json.load(archive.extractfile('snapshot.json'))
        for campaign in snapshot['campaigns']:
            for record in campaign['records']:
                raw = archive.extractfile(f"{campaign['name']}/studies/{record['id']}/comparison.json").read()
                assert hashlib.sha256(raw).hexdigest() == record['source_sha256']['comparison.json']
                report = json.loads(raw)
                if report['status'] != 'complete':
                    continue
                for s in report['students']:
                    pair = s['paired_accuracy']
                    n = s['baseline']['samples']
                    assert n == s['candidate']['samples']
                    broken, fixed = pair['baseline_only_correct'], pair['candidate_only_correct']
                    correct = round(s['baseline']['accuracy'] * n)
                    both_correct = correct - broken
                    both_wrong = n - both_correct - broken - fixed
                    assert min(both_correct, both_wrong, broken, fixed) >= 0
                    assert round(s['candidate']['accuracy']*n) == both_correct + fixed
                    gain = 100 * (fixed-broken)/n
                    assert abs(gain-100*s['accuracy_delta']) < 1e-10
                    p = float(binomtest(fixed, fixed+broken, .5).pvalue) if fixed+broken else 1.
                    assert abs(p-pair['mcnemar_exact_two_sided_p']) < 1e-10
                    ci = pair['paired_bootstrap_95']
                    rows.append({'arm':record['id'], 'seed':s['seed'], 'test_images':n,
                        'both_correct':both_correct, 'both_wrong':both_wrong,
                        'fixed':fixed, 'broken':broken, 'net_fixed':fixed-broken,
                        'gain_pp':gain, 'changed_correctness_pct':100*(fixed+broken)/n,
                        'fixed_fraction_of_discordant':fixed/(fixed+broken) if fixed+broken else None,
                        'baseline_errors_corrected_pct':100*fixed/(n-correct) if n>correct else None,
                        'baseline_correct_broken_pct':100*broken/correct if correct else None,
                        'bootstrap95_low_pp':100*ci[0], 'bootstrap95_high_pp':100*ci[1],
                        'mcnemar_p':p})
    assert len(rows) == 144
    for r, adjusted in zip(rows, holm([x['mcnemar_p'] for x in rows])):
        r['holm_p_144_tests'] = adjusted
        r['evidence'] = ('Improvement' if r['net_fixed'] > 0 else 'Harm') if adjusted < .05 else 'Inconclusive'
    rows.sort(key=lambda r: (r['arm'],r['seed']))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['arm']].append(r)
    summary=[]
    for arm, pairs in grouped.items():
        assert [r['seed'] for r in pairs] == [42,43,44]
        summary.append({'arm':arm,'positive_seeds':sum(r['net_fixed']>0 for r in pairs),
            'negative_seeds':sum(r['net_fixed']<0 for r in pairs),
            'significant_improvement_seeds':sum(r['evidence']=='Improvement' for r in pairs),
            'significant_harm_seeds':sum(r['evidence']=='Harm' for r in pairs),
            'mean_gain_pp':sum(r['gain_pp'] for r in pairs)/3})
    for name,data in [('per-image-pair-analysis.csv',rows),('per-image-arm-summary.csv',summary)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    receipt={'snapshot_utc':snapshot['captured_utc'],'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'pairs':len(rows),'accepted_arms':len(grouped),
        'significant_improvement_pairs':sum(r['evidence']=='Improvement' for r in rows),
        'significant_harm_pairs':sum(r['evidence']=='Harm' for r in rows),
        'inconclusive_pairs':sum(r['evidence']=='Inconclusive' for r in rows),
        'arms_with_all_three_significant_improvements':[r['arm'] for r in summary if r['significant_improvement_seeds']==3],
        'arms_with_at_least_two_significant_improvements':[r['arm'] for r in summary if r['significant_improvement_seeds']>=2],
        'all_correctness_tables_reconstructed_and_checked':True}
    (OUT/'per-image-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# Per-image evidence: each CE-KD versus repaired-KD student pair','',
        f"Snapshot: {snapshot['captured_utc']}. All 144 fitted student pairs from the 48 accepted repairs are included, including negative and unchanged results. Architecture: adapted pretrained ResNet-34 teacher → ResNet-18 student. Each pair shares its student seed and test examples.",'',
        '**Fixed** = CE-KD student wrong, repaired-teacher KD student correct. **Broken** = CE-KD correct, repaired-teacher KD wrong. Net = fixed − broken; accuracy gain = 100 × net / test images. Both-wrong includes wrong-to-different-wrong outcomes; this analysis does not measure probability/rank changes among those outcomes.','',
        'Two-sided exact McNemar tests compare fixed versus broken counts. Holm correction uses all 144 tests together, including harms. A significant row provides conditional evidence for its fitted model pair on test images under the test assumptions; it does not establish significance across student seeds or teacher retraining. Models share images and CE controls, so do not pool counts across seeds or treat policies as independent replications. If test images are correlated within patients/scenes, a cluster-aware image analysis would be needed.','',
        f"Corrected results: **{receipt['significant_improvement_pairs']} improvement pairs**, **{receipt['significant_harm_pairs']} harm pairs**, and **{receipt['inconclusive_pairs']} inconclusive pairs.** Inconclusive does not mean equal performance.",'',
        '## Across-seed pattern for every accepted arm','',
        'Gain columns show seed 42/43/44, in percentage points. `+*` = significant improvement and `−*` = significant harm after Holm correction; an unmarked value is inconclusive. Consistency counts are descriptive, not a new combined hypothesis test.','',
        '| Arm | Seed 42 gain pp | Seed 43 gain pp | Seed 44 gain pp | Positive seeds | Significant gains | Significant harms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        cells=[f"{r['gain_pp']:+.3f}"+(' +*' if r['evidence']=='Improvement' else ' −*' if r['evidence']=='Harm' else '') for r in grouped[s['arm']]]
        lines.append('| '+' | '.join([s['arm'],*cells,f"{s['positive_seeds']}/3",f"{s['significant_improvement_seeds']}/3",f"{s['significant_harm_seeds']}/3"])+' |')
    lines += ['', '## Pair-by-pair correctness evidence','',
        'The CSV also contains both-correct, both-wrong, correctness turnover, the fraction of original errors fixed, original correct predictions broken, raw p-values and the previously saved unadjusted paired-bootstrap intervals. The four correctness cells sum to the original test-set size for every pair.','',
        '| Arm | Seed | Images | Fixed | Broken | Net | Gain pp | Holm p | Evidence |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['test_images']} | {r['fixed']} | {r['broken']} | {r['net_fixed']:+d} | {r['gain_pp']:+.3f} | {r['holm_p_144_tests']:.3g} | {r['evidence']} |")
    lines += ['', '## Interpretation and next steps','',
        'CIFAR-100 LT-IF100 strict is the clearest conditional signal: all three student pairs improve significantly after correction. CIFAR-100 LT-IF50 strict and CINIC-10 IF10 expanded / IF100 strict / IF100 expanded have significant gains in two seeds; inspect the third seed before calling them consistent. The table also exposes significant GTSRB IF100 strict harm. Large CINIC-10 test sets can detect small gains, so p-values alone should not rank practical benefit.','',
        'Use these exploratory results to specify a separate confirmation experiment. Keep the existing teacher fixed if teacher replication is outside scope; add paired student seeds with a prespecified sample size and evaluate all declared outcomes. For transfer-mechanism evidence, cross-tabulate held-out teacher-fixed images against student-fixed/broken images using preserved sample IDs, and inspect target versus non-target classes. Current counts establish student accuracy outcomes, not causal repair retention.','',
        'This report uses saved paired correctness counts and existing metrics, not a new raw-image replay. It does not recover or list individual image IDs. No datasets or models were downloaded, and running experiments were unchanged.','',
        '[Full per-pair CSV](../../runs/experiment-results/repair-significance-20261028/per-image-pair-analysis.csv) · [Arm summary CSV](../../runs/experiment-results/repair-significance-20261028/per-image-arm-summary.csv) · [Validation](../../runs/experiment-results/repair-significance-20261028/per-image-validation.json) · [Across-seed tests](repair-significance-20261028.md)','']
    (ROOT/'docs/experiments/repair-per-image-evidence-20261028.md').write_text('\n'.join(lines))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
