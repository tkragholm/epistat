//! `anova.coxph` and `anova.coxphlist` of survival 3.8-12 against
//! [`epistat_survival::cox::anova`], on the lung data and on simulated
//! counting-process data with strata and case weights, Efron and Breslow ties.
//!
//! # Tolerances
//!
//! The fits of both programs stop when the relative change of the log partial
//! likelihood is below 1e-9, so a maximised log likelihood `l` agrees to
//! `1e-9 |l|`, a bound from the stopping rule that Newton's quadratic
//! convergence makes generous. A statistic is twice a difference of two, so it
//! has the bound `2e-9 (|l_a| + |l_b|)`. The p-value is the upper tail of a
//! chi-square at the statistic. Its density is below 0.5 for one degree of
//! freedom beyond the statistic 0.1, and below 0.5 for any larger one, so the
//! p-value has the absolute bound `0.5` times the statistic's. Neither bound
//! is changed after a run. The worst deviation in units of its bound is printed.

use epistat_survival::SurvData;
use epistat_survival::cox::anova::{AnovaTable, anova, anova_terms};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::load_json;
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Table {
    loglik: Vec<f64>,
    chisq: Vec<Option<f64>>,
    df: Vec<Option<usize>>,
    p: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    ties: String,
    entry: bool,
    strata: bool,
    weights: bool,
    x: Vec<Vec<f64>>,
    groups: Vec<Vec<usize>>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    single: Table,
    single_definition: Table,
    multi: Table,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

fn compare(name: &str, what: &str, got: &AnovaTable, want: &Table, worst: &mut f64) {
    assert_eq!(got.rows.len(), want.loglik.len(), "{name} {what}");
    for (k, row) in got.rows.iter().enumerate() {
        let l = row.log_likelihood.unwrap();
        let tol_l = 1e-9 * want.loglik[k].abs();
        let dev = (l - want.loglik[k]).abs() / tol_l;
        assert!(
            dev <= 1.0,
            "{name} {what} row {k}: loglik {l} vs {}",
            want.loglik[k]
        );
        *worst = worst.max(dev);
        if k == 0 {
            assert!(row.chisq.is_none() && row.df.is_none() && row.p_value.is_none());
            continue;
        }
        let tol = 2e-9 * (want.loglik[k].abs() + want.loglik[k - 1].abs());
        let chisq = row.chisq.unwrap();
        let want_chisq = want.chisq[k].unwrap();
        let dev = (chisq - want_chisq).abs() / tol;
        assert!(
            dev <= 1.0,
            "{name} {what} row {k}: chisq {chisq} vs {want_chisq}"
        );
        *worst = worst.max(dev);
        assert_eq!(row.df, want.df[k], "{name} {what} row {k}: df");
        let p = row.p_value.unwrap();
        let dev = (p - want.p[k].unwrap()).abs() / (0.5 * tol);
        assert!(
            dev <= 1.0,
            "{name} {what} row {k}: p {p} vs {:?}",
            want.p[k]
        );
        *worst = worst.max(dev);
    }
}

#[test]
fn anova_matches_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_anova.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let (mut worst_multi, mut worst_single) = (0.0f64, 0.0f64);
    for c in &reference.cases {
        let n = c.stop.len();
        let p = c.x[0].len();
        let x = Mat::from_fn(n, p, |i, j| c.x[i][j]);
        let mut builder = SurvData::builder(&c.stop, &c.event);
        if c.entry {
            builder = builder.start(&c.start);
        }
        if c.strata {
            builder = builder.stratum(&c.stratum);
        }
        if c.weights {
            builder = builder.weight(&c.weight);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if c.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            ..CoxOptions::default()
        };
        let groups: Vec<&[usize]> = c.groups.iter().map(Vec::as_slice).collect();

        // anova(fit1, ..., fitm) on the nested prefixes.
        let mut fits = Vec::new();
        let mut columns: Vec<usize> = Vec::new();
        for g in &groups {
            columns.extend_from_slice(g);
            let sub = Mat::from_fn(n, columns.len(), |i, j| c.x[i][columns[j]]);
            fits.push(fit(&data, sub.as_ref(), &opts).unwrap());
        }
        let refs: Vec<&_> = fits.iter().collect();
        let table = anova(&refs).unwrap();
        compare(&c.name, "list", &table, &c.multi, &mut worst_multi);
        assert_eq!(table.n_rows, n);

        // anova(fit): the null model and the terms in turn. R refits without
        // the weights, so the weighted case is compared with the definition.
        let single = anova_terms(&data, x.as_ref(), &groups, &opts).unwrap();
        let want = if c.weights {
            &c.single_definition
        } else {
            &c.single
        };
        compare(&c.name, "single", &single, want, &mut worst_single);
        assert_eq!(single.rows.len(), groups.len() + 1);
        if c.weights {
            let differs = (c.single.loglik[1] - c.single_definition.loglik[1]).abs() > 1.0;
            assert!(differs, "{}: R's refits should drop the weights", c.name);
        }
    }
    eprintln!("worst deviation, list: {worst_multi:.2e} of its bound");
    eprintln!("worst deviation, single: {worst_single:.2e} of its bound");
}
