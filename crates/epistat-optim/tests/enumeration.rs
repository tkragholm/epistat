//! Branch and bound, the dense simplex method and the network simplex method
//! against enumeration and reference solvers.
//!
//! Random small programs have every integer point enumerated and the optimum
//! compared with the solver's, with and without root heuristics and under node
//! limits, where the bound must stay above the optimum. Flows are compared with
//! successive shortest paths. The dense simplex method must judge infeasibility
//! the same at any scale of the data.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    reason = "small test programs index by position and convert between counts and values \
              that are far below the limits of the types"
)]

use epistat_optim::branch_and_bound::{MipError, MixedBinaryProgram, Options, Status, solve};
use epistat_optim::min_cost_flow::{FlowError, MinCostFlow};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % ((hi - lo + 1) as u64)) as i64
    }
}

const INF: f64 = f64::INFINITY;

/// Exhaustive optimum: every whole column enumerated within its bounds, and the
/// last column continuous when `continuous` (rows then give an interval for it,
/// and its cost picks the end).
fn brute(
    m: usize,
    columns: &[f64],
    cost: &[f64],
    lo: &[f64],
    hi: &[f64],
    row_lo: &[f64],
    row_hi: &[f64],
    continuous: bool,
) -> Option<f64> {
    let n = cost.len();
    let nw = if continuous { n - 1 } else { n };
    let mut best: Option<f64> = None;
    let mut x: Vec<f64> = (0..nw).map(|j| lo[j]).collect();
    loop {
        let mut lo_t = if continuous { lo[n - 1] } else { 0.0 };
        let mut hi_t = if continuous { hi[n - 1] } else { 0.0 };
        let mut ok = true;
        for i in 0..m {
            let s: f64 = (0..nw).map(|j| columns[j * m + i] * x[j]).sum();
            let b = if continuous {
                columns[(n - 1) * m + i]
            } else {
                0.0
            };
            if b == 0.0 {
                if s < row_lo[i] - 1e-9 || s > row_hi[i] + 1e-9 {
                    ok = false;
                }
            } else {
                let (u, v) = ((row_lo[i] - s) / b, (row_hi[i] - s) / b);
                let (u, v) = if b > 0.0 { (u, v) } else { (v, u) };
                lo_t = lo_t.max(u);
                hi_t = hi_t.min(v);
            }
        }
        if ok && lo_t <= hi_t + 1e-9 {
            let mut value: f64 = (0..nw).map(|j| cost[j] * x[j]).sum();
            if continuous {
                let c = cost[n - 1];
                value += if c > 0.0 {
                    c * hi_t
                } else if c < 0.0 {
                    c * lo_t
                } else {
                    0.0
                };
            }
            if best.is_none_or(|b| value > b) {
                best = Some(value);
            }
        }
        // next point
        let mut j = 0;
        loop {
            if j == nw {
                return best;
            }
            if x[j] < hi[j] {
                x[j] += 1.0;
                break;
            }
            x[j] = lo[j];
            j += 1;
        }
    }
}

fn check_trial(
    trial: usize,
    m: usize,
    columns: &[f64],
    cost: &[f64],
    lo: &[f64],
    hi: &[f64],
    row_lo: &[f64],
    row_hi: &[f64],
    binary: &[bool],
    integer: &[bool],
    continuous: bool,
    options: &Options,
) -> Result<(), String> {
    let program = MixedBinaryProgram {
        rows: m,
        columns,
        cost,
        col_lo: lo,
        col_hi: hi,
        row_lo,
        row_hi,
        binary,
        integer,
    };
    let reference = brute(m, columns, cost, lo, hi, row_lo, row_hi, continuous);
    let got = solve(&program, options);
    match (reference, got) {
        (Some(r), Ok(s)) => {
            if s.status != Status::Optimal {
                return Err(format!("trial {trial}: status {:?}", s.status));
            }
            if (s.objective - r).abs() > 1e-7 * r.abs().max(1.0) {
                return Err(format!(
                    "trial {trial}: objective {} but the optimum is {r}; bound {}",
                    s.objective, s.bound
                ));
            }
            if s.bound < r - 1e-7 * r.abs().max(1.0) {
                return Err(format!(
                    "trial {trial}: bound {} below the optimum {r}",
                    s.bound
                ));
            }
            if s.max_row_violation > 1e-6 {
                return Err(format!("trial {trial}: violation {}", s.max_row_violation));
            }
            Ok(())
        }
        (None, Err(MipError::Infeasible)) => Ok(()),
        (r, g) => Err(format!("trial {trial}: reference {r:?}, solver {g:?}")),
    }
}

/// Knapsack-like programs with integer coefficients, so that the row slacks are
/// whole and the Gomory cuts use them, with equality and two-sided rows,
/// negative costs, general integer columns and columns fixed at 1.
#[test]
fn integer_programs_agree_with_enumeration() {
    let mut rng = Rng(77);
    let mut failures = Vec::new();
    for trial in 0..3000 {
        let nbin = 4 + trial % 9;
        let ngen = trial % 3;
        let n = nbin + ngen;
        let m = 1 + trial % 4;
        let mut columns = vec![0.0; n * m];
        for v in &mut columns {
            *v = rng.int(-3, 6) as f64;
        }
        let cost: Vec<f64> = (0..n).map(|_| rng.int(-2, 9) as f64).collect();
        let mut lo = vec![0.0; n];
        let mut hi = vec![1.0; n];
        for j in 0..nbin {
            if rng.int(0, 9) == 0 {
                lo[j] = 1.0;
            }
        }
        for j in nbin..n {
            lo[j] = rng.int(-2, 0) as f64;
            hi[j] = rng.int(1, 3) as f64;
        }
        let mut row_lo = vec![-INF; m];
        let mut row_hi = vec![INF; m];
        for i in 0..m {
            let kind = rng.int(0, 3);
            let centre = rng.int(0, (2 * n) as i64) as f64;
            match kind {
                0 => row_hi[i] = centre,
                1 => row_lo[i] = centre,
                2 => {
                    row_lo[i] = centre;
                    row_hi[i] = centre;
                }
                _ => {
                    row_lo[i] = centre - rng.int(0, 3) as f64;
                    row_hi[i] = centre + rng.int(0, 3) as f64;
                }
            }
        }
        let binary: Vec<bool> = (0..n).map(|j| j < nbin).collect();
        let integer: Vec<bool> = vec![true; n];
        if let Err(e) = check_trial(
            trial,
            m,
            &columns,
            &cost,
            &lo,
            &hi,
            &row_lo,
            &row_hi,
            &binary,
            &integer,
            false,
            &Options::default(),
        ) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

/// Real coefficients and costs (no objective step), with one continuous column
/// that has a cost, and binary columns only otherwise.
#[test]
fn real_programs_agree_with_enumeration() {
    let mut rng = Rng(91);
    let mut failures = Vec::new();
    for trial in 0..3000 {
        let nbin = 4 + trial % 10;
        let n = nbin + 1;
        let m = 1 + trial % 4;
        let mut columns = vec![0.0; n * m];
        for v in &mut columns {
            *v = rng.uniform(-1.0, 2.0);
        }
        let mut cost: Vec<f64> = (0..n).map(|_| rng.uniform(-0.5, 2.0)).collect();
        if trial % 2 == 0 {
            cost[nbin] = 0.0;
        }
        let mut lo = vec![0.0; n];
        let mut hi = vec![1.0; n];
        lo[nbin] = 0.0;
        hi[nbin] = rng.uniform(0.5, 4.0);
        let mut row_lo = vec![-INF; m];
        let mut row_hi = vec![INF; m];
        for i in 0..m {
            let centre = rng.uniform(0.0, nbin as f64 / 2.0);
            match rng.int(0, 2) {
                0 => row_hi[i] = centre,
                1 => row_lo[i] = centre - 1.0,
                _ => {
                    row_lo[i] = centre - 0.5;
                    row_hi[i] = centre + 0.5;
                }
            }
        }
        let binary: Vec<bool> = (0..n).map(|j| j < nbin).collect();
        if let Err(e) = check_trial(
            trial,
            m,
            &columns,
            &cost,
            &lo,
            &hi,
            &row_lo,
            &row_hi,
            &binary,
            &[],
            true,
            &Options::default(),
        ) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

/// Same integer programs without the root heuristics, so the levels and the
/// branching carry the proof alone.
#[test]
fn integer_programs_without_heuristics_agree_with_enumeration() {
    let mut rng = Rng(1234);
    let mut failures = Vec::new();
    let options = Options {
        neighbourhood_search: false,
        ..Options::default()
    };
    for trial in 0..2000 {
        let nbin = 5 + trial % 8;
        let n = nbin;
        let m = 1 + trial % 3;
        let mut columns = vec![0.0; n * m];
        for v in &mut columns {
            *v = rng.int(0, 7) as f64;
        }
        let cost: Vec<f64> = (0..n).map(|_| rng.int(1, 12) as f64).collect();
        let lo = vec![0.0; n];
        let hi = vec![1.0; n];
        let row_lo = vec![-INF; m];
        let row_hi: Vec<f64> = (0..m).map(|_| rng.int(3, 3 * n as i64) as f64).collect();
        let binary = vec![true; n];
        if let Err(e) = check_trial(
            trial,
            m,
            &columns,
            &cost,
            &lo,
            &hi,
            &row_lo,
            &row_hi,
            &binary,
            &[],
            false,
            &options,
        ) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

// ---------------------------------------------------------------------------
// Minimum-cost flow against successive shortest paths by Bellman-Ford.

fn reference_flow(nodes: usize, supply: &[i64], arcs: &[(usize, usize, i64, i64)]) -> Option<i128> {
    // Super source and sink; negative costs handled by Bellman-Ford on the
    // residual graph (no negative cycles in the generated instances).
    let s = nodes;
    let t = nodes + 1;
    let total = nodes + 2;
    let mut to = Vec::new();
    let mut cap = Vec::new();
    let mut cost = Vec::new();
    let mut adj = vec![Vec::new(); total];
    let add = |a: usize,
               b: usize,
               c: i64,
               w: i64,
               to: &mut Vec<usize>,
               cap: &mut Vec<i64>,
               cost: &mut Vec<i64>,
               adj: &mut Vec<Vec<usize>>| {
        adj[a].push(to.len());
        to.push(b);
        cap.push(c);
        cost.push(w);
        adj[b].push(to.len());
        to.push(a);
        cap.push(0);
        cost.push(-w);
    };
    let mut need = 0;
    for &(a, b, c, w) in arcs {
        add(a, b, c, w, &mut to, &mut cap, &mut cost, &mut adj);
    }
    for v in 0..nodes {
        if supply[v] > 0 {
            add(s, v, supply[v], 0, &mut to, &mut cap, &mut cost, &mut adj);
            need += supply[v];
        } else if supply[v] < 0 {
            add(v, t, -supply[v], 0, &mut to, &mut cap, &mut cost, &mut adj);
        }
    }
    let mut flow = 0;
    let mut total_cost: i128 = 0;
    while flow < need {
        let mut dist = vec![i64::MAX; total];
        let mut prev = vec![usize::MAX; total];
        dist[s] = 0;
        for _ in 0..total {
            let mut changed = false;
            for v in 0..total {
                if dist[v] == i64::MAX {
                    continue;
                }
                for &e in &adj[v] {
                    if cap[e] > 0 && dist[v] + cost[e] < dist[to[e]] {
                        dist[to[e]] = dist[v] + cost[e];
                        prev[to[e]] = e;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if dist[t] == i64::MAX {
            return None;
        }
        let mut push = need - flow;
        let mut v = t;
        while v != s {
            let e = prev[v];
            push = push.min(cap[e]);
            v = to[e ^ 1];
        }
        let mut v = t;
        while v != s {
            let e = prev[v];
            cap[e] -= push;
            cap[e ^ 1] += push;
            v = to[e ^ 1];
        }
        flow += push;
        total_cost += i128::from(push) * i128::from(dist[t]);
    }
    Some(total_cost)
}

#[test]
fn network_simplex_agrees_with_successive_shortest_paths() {
    let mut rng = Rng(5);
    let mut failures = Vec::new();
    for trial in 0..3000 {
        let nodes = 3 + trial % 10;
        let narcs = nodes + rng.int(0, 3 * nodes as i64) as usize;
        let mut arcs = Vec::new();
        for _ in 0..narcs {
            let a = rng.int(0, nodes as i64 - 1) as usize;
            let mut b = rng.int(0, nodes as i64 - 1) as usize;
            if a == b {
                b = (b + 1) % nodes;
            }
            // Non-negative costs (no negative cycles), zero capacities included.
            arcs.push((a, b, rng.int(0, 4), rng.int(0, 9)));
        }
        let mut supply = vec![0_i64; nodes];
        for _ in 0..rng.int(1, 4) {
            let a = rng.int(0, nodes as i64 - 1) as usize;
            let b = rng.int(0, nodes as i64 - 1) as usize;
            let q = rng.int(1, 3);
            supply[a] += q;
            supply[b] -= q;
        }
        let mut problem = MinCostFlow::new(nodes);
        for (v, &q) in supply.iter().enumerate() {
            problem.set_supply(v, q);
        }
        for &(a, b, c, w) in &arcs {
            problem.add_arc(a, b, c, w);
        }
        let reference = reference_flow(nodes, &supply, &arcs);
        match (reference, problem.solve()) {
            (Some(r), Ok(s)) => {
                if s.cost != r {
                    failures.push(format!("trial {trial}: cost {} vs {r}", s.cost));
                }
                let mut balance = supply.clone();
                for (k, &(a, b, c, w)) in arcs.iter().enumerate() {
                    let f = s.flow[k];
                    if f < 0 || f > c {
                        failures.push(format!("trial {trial}: arc {k} flow {f} cap {c}"));
                    }
                    balance[a] -= f;
                    balance[b] += f;
                    let red = w + s.potentials[a] - s.potentials[b];
                    if c > 0
                        && ((f == 0 && red < 0)
                            || (f == c && red > 0)
                            || (f > 0 && f < c && red != 0))
                    {
                        failures.push(format!(
                            "trial {trial}: arc {k} reduced cost {red} at flow {f}/{c}"
                        ));
                    }
                }
                if balance.iter().any(|b| *b != 0) {
                    failures.push(format!("trial {trial}: conservation"));
                }
            }
            (None, Err(FlowError::Infeasible)) => {}
            (r, g) => failures.push(format!("trial {trial}: reference {r:?}, solver {g:?}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures, first: {:?}",
        failures.len(),
        &failures[..failures.len().min(5)]
    );
}

/// `λ ≥ 0` with `λ = -1e-9` has no solution; the phase-one tolerance is an
/// absolute `1e-8`, so a program on a small scale is reported solved.
#[test]
fn dense_simplex_reports_an_infeasible_program_on_a_small_scale() {
    use epistat_optim::simplex::{DenseSimplex, LpError};
    let mut lp = DenseSimplex::new(&[vec![1.0]], &[-1e-9]);
    lp.set_cost(&[1.0]);
    let result = lp.solve();
    assert_eq!(
        result,
        Err(LpError::Infeasible),
        "solved with lambda {:?} and objective {}",
        lp.lambda(),
        lp.objective()
    );
}

/// The same program scaled by 1e9 is reported infeasible.
#[test]
fn dense_simplex_reports_the_same_program_at_unit_scale_infeasible() {
    use epistat_optim::simplex::{DenseSimplex, LpError};
    let mut lp = DenseSimplex::new(&[vec![1.0]], &[-1.0]);
    lp.set_cost(&[1.0]);
    assert_eq!(lp.solve(), Err(LpError::Infeasible));
}

/// A circulation of negative cost with capacities far above `i64::MAX / 4`.
#[test]
fn a_huge_capacity_negative_cycle_is_solved_exactly() {
    let mut problem = MinCostFlow::new(3);
    problem.add_arc(0, 1, i64::MAX, -1);
    problem.add_arc(1, 2, i64::MAX, -1);
    problem.add_arc(2, 0, i64::MAX, -1);
    let s = problem.solve().expect("a solution");
    assert_eq!(s.flow, vec![i64::MAX; 3]);
    assert_eq!(s.cost, -3 * i128::from(i64::MAX));
}

/// Under a node limit the reported bound must still be at least the optimum,
/// and a status of Optimal must mean the optimum.
#[test]
fn bounds_under_node_limits_hold() {
    let mut rng = Rng(4242);
    let mut failures = Vec::new();
    for trial in 0..1500 {
        let nbin = 8 + trial % 9;
        let integer_data = trial % 2 == 0;
        let n = nbin + usize::from(!integer_data);
        let m = 1 + trial % 4;
        let mut columns = vec![0.0; n * m];
        for v in &mut columns {
            *v = if integer_data {
                rng.int(-2, 7) as f64
            } else {
                rng.uniform(-1.0, 2.0)
            };
        }
        let cost: Vec<f64> = (0..n)
            .map(|_| {
                if integer_data {
                    rng.int(-1, 9) as f64
                } else {
                    rng.uniform(-0.3, 2.0)
                }
            })
            .collect();
        let lo = vec![0.0; n];
        let mut hi = vec![1.0; n];
        if !integer_data {
            hi[nbin] = 2.5;
        }
        let row_lo = vec![-INF; m];
        let row_hi: Vec<f64> = (0..m)
            .map(|_| {
                if integer_data {
                    rng.int(2, 2 * nbin as i64) as f64
                } else {
                    rng.uniform(1.0, nbin as f64 / 2.0)
                }
            })
            .collect();
        let binary: Vec<bool> = (0..n).map(|j| j < nbin).collect();
        let program = MixedBinaryProgram {
            rows: m,
            columns: &columns,
            cost: &cost,
            col_lo: &lo,
            col_hi: &hi,
            row_lo: &row_lo,
            row_hi: &row_hi,
            binary: &binary,
            integer: &[],
        };
        let Some(optimum) = brute(
            m,
            &columns,
            &cost,
            &lo,
            &hi,
            &row_lo,
            &row_hi,
            !integer_data,
        ) else {
            continue;
        };
        for nodes in [1_u64, 2, 3, 5, 8, 13] {
            for heuristics in [false, true] {
                let options = Options {
                    node_limit: Some(nodes),
                    time_limit: None,
                    neighbourhood_search: heuristics,
                };
                let tol = 1e-7 * optimum.abs().max(1.0);
                match solve(&program, &options) {
                    Ok(s) => {
                        if s.bound < optimum - tol {
                            failures.push(format!(
                                "trial {trial} nodes {nodes}: bound {} below optimum {optimum}",
                                s.bound
                            ));
                        }
                        if s.objective > optimum + tol {
                            failures.push(format!(
                                "trial {trial} nodes {nodes}: objective {} above optimum {optimum}",
                                s.objective
                            ));
                        }
                        if s.status == Status::Optimal && (s.objective - optimum).abs() > tol {
                            failures.push(format!(
                                "trial {trial} nodes {nodes}: Optimal at {} but optimum {optimum}",
                                s.objective
                            ));
                        }
                        if s.status != Status::Optimal
                            && s.gap() == 0.0
                            && (s.objective - optimum).abs() > tol
                        {
                            failures.push(format!(
                                "trial {trial} nodes {nodes}: zero gap at {} but optimum {optimum}",
                                s.objective
                            ));
                        }
                    }
                    Err(MipError::NoSolution { bound, .. }) => {
                        if bound < optimum - tol {
                            failures.push(format!("trial {trial} nodes {nodes}: no solution, bound {bound} below optimum {optimum}"));
                        }
                    }
                    Err(e) => failures.push(format!("trial {trial} nodes {nodes}: {e}")),
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures: {:#?}",
        failures.len(),
        &failures[..failures.len().min(6)]
    );
}

/// Larger programs of the matching shape: two count rows tied to an integer
/// size column `k` with a ratio, and balance rows with real coefficients.
#[test]
fn matching_shaped_programs_with_a_ratio_agree_with_enumeration() {
    let mut rng = Rng(99);
    let mut failures = Vec::new();
    for trial in 0..400 {
        let nbin = 12 + trial % 7;
        let ratio = 1.0 + (trial % 3) as f64;
        let p = 1 + trial % 3;
        let m = 2 + 2 * p;
        let n = nbin + 1;
        let focal: Vec<bool> = (0..nbin).map(|j| j % 3 == 0).collect();
        let mut columns = vec![0.0; n * m];
        for j in 0..nbin {
            columns[j * m + usize::from(!focal[j])] = 1.0;
        }
        columns[nbin * m] = -1.0;
        columns[nbin * m + 1] = -ratio;
        let tol = rng.uniform(0.05, 0.5);
        for c in 0..p {
            for j in 0..nbin {
                let v = rng.uniform(0.0, 2.0) + if focal[j] { 0.4 } else { 0.0 };
                let coef = if focal[j] { v } else { -v / ratio };
                columns[j * m + 2 + c] = coef;
                columns[j * m + 2 + p + c] = coef;
            }
            columns[nbin * m + 2 + c] = -tol;
            columns[nbin * m + 2 + p + c] = tol;
        }
        let mut row_lo = vec![0.0; m];
        let mut row_hi = vec![0.0; m];
        for c in 0..p {
            row_lo[2 + c] = -INF;
            row_hi[2 + p + c] = INF;
        }
        let mut cost = vec![1.0; n];
        cost[nbin] = 0.0;
        let lo = vec![0.0; n];
        let mut hi = vec![1.0; n];
        hi[nbin] = focal.iter().filter(|f| **f).count() as f64;
        let binary: Vec<bool> = (0..n).map(|j| j < nbin).collect();
        let integer: Vec<bool> = (0..n).map(|j| j == nbin).collect();
        if let Err(e) = check_trial(
            trial,
            m,
            &columns,
            &cost,
            &lo,
            &hi,
            &row_lo,
            &row_hi,
            &binary,
            &integer,
            false,
            &Options::default(),
        ) {
            failures.push(e);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures: {:#?}",
        failures.len(),
        &failures[..failures.len().min(6)]
    );
}
