"""Exploratory paired-seed and conditional per-image tests of accepted repairs."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import tarfile

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/experiment-results/current-results-20261027/source-metadata.tar.gz'
OUT = ROOT / 'runs/experiment-results/repair-significance-20261028'


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.] * len(values)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(values) - rank) * values[i]))
        adjusted[i] = running
    return adjusted


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, images = [], []
    with tarfile.open(SOURCE) as archive:
        snapshot = json.load(archive.extractfile('snapshot.json'))
        for campaign in snapshot['campaigns']:
            for record in campaign['records']:
                raw = archive.extractfile(f"{campaign['name']}/studies/{record['id']}/comparison.json").read()
                assert hashlib.sha256(raw).hexdigest() == record['source_sha256']['comparison.json']
                report = json.loads(raw)
                if report['status'] != 'complete':
                    continue
                students = report['students']
                assert [s['seed'] for s in students] == [42, 43, 44]
                differences = np.array([s['candidate']['accuracy'] - s['baseline']['accuracy'] for s in students])
                assert np.allclose(differences, report['aggregate']['accuracy_delta']['values'], atol=1e-12, rtol=0)
                assert differences.std(ddof=1) > 0
                test = stats.ttest_1samp(differences, 0.)
                paired = stats.ttest_rel([s['candidate']['accuracy'] for s in students], [s['baseline']['accuracy'] for s in students])
                assert abs(test.pvalue - paired.pvalue) < 1e-12
                ci = test.confidence_interval(.95)
                perm = [abs(np.mean(differences * signs)) for signs in itertools.product([-1, 1], repeat=3)]
                exact_p = float(sum(x >= abs(differences.mean()) - 1e-14 for x in perm) / len(perm))
                rows.append({'arm': record['id'], 'mean_gain_pp':100*differences.mean(),
                    'seed42_gain_pp':100*differences[0], 'seed43_gain_pp':100*differences[1],
                    'seed44_gain_pp':100*differences[2], 't_ci95_low_pp':100*ci.low,
                    't_ci95_high_pp':100*ci.high, 'paired_t_p':float(test.pvalue),
                    'exact_sign_flip_p':exact_p})
                for student in students:
                    p = student['paired_accuracy']
                    b, c = p['baseline_only_correct'], p['candidate_only_correct']
                    mcnemar = stats.binomtest(c, b+c, .5).pvalue if b+c else 1.
                    assert abs(mcnemar-p['mcnemar_exact_two_sided_p']) < 1e-10
                    images.append({'arm':record['id'], 'seed':student['seed'],
                        'gain_pp':100*(student['candidate']['accuracy']-student['baseline']['accuracy']),
                        'baseline_only_correct':b, 'repair_only_correct':c, 'mcnemar_p':float(mcnemar)})
    assert len(rows) == 48 and len({x['arm'] for x in rows}) == 48
    for row, adjusted in zip(rows, holm([x['paired_t_p'] for x in rows])):
        row['paired_t_holm_p'] = adjusted
    for row, adjusted in zip(images, holm([x['mcnemar_p'] for x in images])):
        row['mcnemar_holm_p'] = adjusted
    rows.sort(key=lambda x: -x['mean_gain_pp'])
    save_csv(OUT/'paired-seed-tests.csv', rows)
    save_csv(OUT/'conditional-image-tests.csv', images)
    positive = [x for x in rows if x['mean_gain_pp'] > 1e-8]
    summary = {'source_snapshot_utc':snapshot['captured_utc'],
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'accepted_arms':len(rows),'positive_mean_arms':len(positive),
        'positive_nominal_t_p_below_005':sum(x['paired_t_p'] < .05 for x in positive),
        'positive_holm_t_p_below_005':sum(x['paired_t_holm_p'] < .05 for x in positive),
        'positive_exact_sign_flip_p_below_005':sum(x['exact_sign_flip_p'] < .05 for x in positive),
        'conditional_image_tests':len(images),
        'positive_conditional_mcnemar_holm_p_below_005':sum(x['gain_pp']>0 and x['mcnemar_holm_p']<.05 for x in images),
        'all_tests_two_sided':True, 'alpha':.05,
        'seed_test_family':48,'image_test_family':144,
        'scope':'Exploratory; one fixed teacher per setting, three paired student seeds; rejected repairs have no new student comparisons.'}
    (OUT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Repair versus paired CE-KD: exploratory statistical tests','',
        f"Source snapshot: {snapshot['captured_utc']}. All 48 accepted repair arms are tested; the table shows the 32 positive means. Negative and unchanged arms remain in the multiplicity family.",'',
        'Student-seed analysis uses three paired seeds (42/43/44), two-sided paired t-tests, unadjusted 95% t intervals, and Holm adjustment over 48 accuracy tests. The exact paired sign-flip test enumerates all eight sign assignments; its smallest possible two-sided p-value is 0.25. Neither test generalizes over teacher fitting seeds. With n=3 the normality assumption for t-tests cannot be assessed reliably, so parametric results are tentative. Shared controls/policies are dependent; Holm accommodates dependence, but these are not independent teacher replications.','',
        f"Positive means: {len(positive)}. Nominal t-test p<0.05: {summary['positive_nominal_t_p_below_005']}; Holm-adjusted: {summary['positive_holm_t_p_below_005']}; exact sign-flip: {summary['positive_exact_sign_flip_p_below_005']}.",'',
        '| Repair arm | Mean gain pp | Unadjusted 95% t CI pp | t p | Holm p (48 tests) | Exact sign-flip p |',
        '|---|---:|---|---:|---:|---:|']
    for x in positive:
        lines.append(f"| {x['arm']} | {x['mean_gain_pp']:+.3f} | [{x['t_ci95_low_pp']:+.3f}, {x['t_ci95_high_pp']:+.3f}] | {x['paired_t_p']:.4f} | {x['paired_t_holm_p']:.4f} | {x['exact_sign_flip_p']:.2f} |")
    lines += ['', 'The separate CSV contains exact McNemar tests from saved discordant per-image correctness counts for all 144 fitted student pairs, corrected together with Holm. These measure image-level evidence conditional on each fitted pair; do not pool seeds as independent test images. They do not establish reliable gains across training seeds. No new predictions or models were trained.', '',
        'The CI column is unadjusted and does not provide simultaneous 95% coverage across arms. Target recall/F1 significance is outside this accuracy-only analysis. Practical success and statistical significance are separate.', '',
        'Next: prespecify a confirmation setting, budget, metric and test before further inspection; use additional independently paired student seeds (at least 10 permits stronger assessment, but choose sample size via power analysis). Teacher-fixed conclusions remain conditional on that teacher. Evaluate all prespecified outcomes, including negative ones. Existing test-set reuse makes this exploratory.', '',
        'Methods: [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html), [SciPy exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [Holm correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).', '',
        '[All paired-seed tests](../../runs/experiment-results/repair-significance-20261028/paired-seed-tests.csv) · [Conditional per-image tests](../../runs/experiment-results/repair-significance-20261028/conditional-image-tests.csv) · [Validation](../../runs/experiment-results/repair-significance-20261028/validation.json)', '']
    (ROOT/'docs/experiments/repair-significance-20261028.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
