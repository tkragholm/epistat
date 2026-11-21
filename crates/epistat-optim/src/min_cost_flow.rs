//! Minimum-cost flow by the primal network simplex method.
//!
//! Solves
//!
//! ```text
//! min  sum_a c_a x_a
//! s.t. sum_{a leaves v} x_a - sum_{a enters v} x_a = b_v   for every node v
//!      0 <= x_a <= u_a                                    for every arc a
//! ```
//!
//! with integer costs `c_a`, integer capacities `u_a` and integer supplies
//! `b_v` (a node with `b_v > 0` supplies flow, one with `b_v < 0` demands it),
//! `sum_v b_v = 0`. Because the data are integers, the solution is integer and
//! the optimum is exact. A problem with no feasible flow is
//! [`FlowError::Infeasible`].
//!
//! # Algorithm
//!
//! The method is the network simplex method of Ahuja, Magnanti and Orlin \[1,
//! ch. 11\], in the form of a spanning tree of the network plus one artificial
//! root. A basis is a spanning tree `T`. Every arc outside the tree is at its
//! lower bound 0 or its upper bound `u_a`, and the flow on the tree arcs is
//! then determined by the supplies. The node potentials `pi` are the ones for
//! which every tree arc has reduced cost
//!
//! ```text
//! c^pi_a = c_a + pi_tail - pi_head = 0.
//! ```
//!
//! Start. Every node `v` is joined to the root by an artificial arc of cost
//! `M`, from `v` when `b_v > 0` and to `v` otherwise, carrying `|b_v|`. All
//! other arcs are at 0. `M` exceeds the cost of any path in the network
//! (`M = (n + 1)(max |c_a| + 1)`), so a feasible problem ends with no flow on
//! the artificial arcs, and any flow left on them means the problem has no
//! feasible flow.
//!
//! Pricing. An arc outside the tree violates optimality when it is at its
//! lower bound with `c^pi < 0`, or at its upper bound with `c^pi > 0`. The
//! entering arc is chosen by block search \[2\]: the arcs are scanned in
//! blocks of `max(sqrt(m), 10)`, starting where the last scan stopped and
//! wrapping round, and the arc of the most violated reduced cost in the first
//! block that has a violation enters. When a full sweep finds none, the flow
//! is optimal.
//!
//! Pivot. The entering arc closes a cycle with the tree. The flow around the
//! cycle changes by the smallest residual capacity of its arcs, and an arc
//! that reaches the limit leaves the tree. When several arcs reach it at once,
//! the leaving arc is the last of them met when the cycle is walked from its
//! apex (the nearest common ancestor of the entering arc's ends) in the
//! direction of the flow change. This is Cunningham's rule \[3\]. It keeps the
//! tree strongly feasible (every tree arc with no flow points away from the
//! root, and every tree arc at capacity points toward it), which makes a
//! degenerate pivot change the potentials, so the method cannot cycle. The
//! subtree below the leaving arc is hung from the entering arc, with its path
//! to the old attachment point reversed, and its potentials and depths are
//! recomputed.
//!
//! # Warm starts
//!
//! [`MinCostFlow::solve_from`] starts from the final tree of an earlier solve
//! of the same nodes and supplies, with arcs added at the end. The new arcs
//! start at their lower bound, the flows and potentials are rebuilt from the
//! tree and the bounds, and the first pricing search starts at the new arcs.
//! The tree is still strongly feasible, since the flow and the tree arcs are
//! those of the earlier optimum, so the guarantees above hold. A problem that
//! is solved with a few arcs, priced against the potentials for arcs that
//! would enter, and solved again with those added, therefore costs the pivots
//! that the new arcs need and not a solve from scratch.
//!
//! [`MinCostFlow::solve_from_flow`] starts from a feasible flow instead. Every
//! arc strictly between its bounds is in the tree, which must not close a
//! cycle, and each component of such arcs hangs from the root by an artificial
//! arc that leaves the root and carries nothing. A tree arc with no flow then
//! points away from the root and every arc at its capacity is outside the
//! tree, which is strong feasibility. A flow that is already near the optimum
//! needs few pivots, and the pivots it needs are the ones that matter, where a
//! start from the artificial arcs spends many on building a flow first.
//!
//! # Pricing block
//!
//! The block of the pricing search can be set with
//! [`MinCostFlow::set_pricing_block`]. A pivot costs the arcs scanned for the
//! entering arc plus the cycle and the re-hung subtree, so when the latter is
//! small a short block is faster in total even though it takes more pivots.
//!
//! # Cost
//!
//! A pivot costs `O(cycle length + size of the re-hung subtree)`, at most
//! `O(n)`, and a pricing sweep `O(m)`. No polynomial bound on the number of
//! pivots holds for this pivot rule, but on assignment and transportation
//! networks of a few thousand nodes it needs a small multiple of `n` pivots in
//! practice. Memory is about 33 bytes per arc plus a few arrays of length `n`.
//!
//! # Ties
//!
//! Several flows can have the same minimum cost. The one returned is a
//! function of the input alone: the arcs are scanned in the order they were
//! added, the first of equally violated arcs in a block enters, and leaving
//! arcs follow Cunningham's rule. It does not depend on threads or on
//! addresses. A caller that wants a particular tie-break adds it to the costs.
//!
//! # Limits
//!
//! Costs and potentials are `i64`. The solver refuses input for which
//! `4 (n + 2)(max |c_a| + 1)` or the sum of the supplies could overflow, with
//! [`FlowError::Overflow`]. A capacity may be any non-negative `i64`, including
//! `i64::MAX`, and a capacity of 0 gives an arc that carries no flow. There are at most `u32::MAX - 1` nodes and `u32::MAX` arcs.
//!
//! # References
//!
//! 1. Ahuja, R. K., Magnanti, T. L. and Orlin, J. B. (1993). *Network Flows:
//!    Theory, Algorithms, and Applications.* Prentice Hall. Chapter 11.
//! 2. Dantzig, G. B. (1963). *Linear Programming and Extensions.* Princeton
//!    University Press. Ch. 17, on the transportation problem; block pricing
//!    is the partial pricing of Ch. 9.
//! 3. Cunningham, W. H. (1976). A network simplex method. *Mathematical
//!    Programming* 11, 105-116.

#![expect(
    clippy::cast_possible_truncation,
    reason = "node and arc counts are checked against u32::MAX before any index is narrowed"
)]

use std::fmt;

const NONE: u32 = u32::MAX;
const LOWER: i8 = 1;
const UPPER: i8 = -1;
const TREE: i8 = 0;

/// Why a flow could not be found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowError {
    /// No flow meets the supplies within the capacities, or the supplies do
    /// not sum to zero.
    Infeasible,
    /// A cost, capacity or supply is too large for 64-bit arithmetic, or
    /// there are more nodes or arcs than the solver indexes.
    Overflow,
    /// An arc names a node that does not exist, or has a negative capacity.
    InvalidInput,
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible => f.write_str("no flow satisfies the supplies within the capacities"),
            Self::Overflow => f.write_str("a cost, capacity or supply is too large to solve"),
            Self::InvalidInput => f.write_str("an arc has an unknown node or a negative capacity"),
        }
    }
}

impl std::error::Error for FlowError {}

/// An optimal flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowSolution {
    /// Flow on each arc, in the order the arcs were added.
    pub flow: Vec<i64>,
    /// The total cost, `sum c_a x_a`.
    pub cost: i128,
    /// Node potentials with `c_a + pi_tail - pi_head >= 0` for every arc at
    /// its lower bound, `<= 0` at its upper bound and `= 0` strictly between.
    pub potentials: Vec<i64>,
    /// Number of pivots.
    pub pivots: u64,
    /// The final basis, for [`MinCostFlow::solve_from`].
    pub basis: Basis,
}

/// The spanning tree and the bound of every arc outside it at the end of a
/// solve. It starts a later solve of the same nodes and supplies with more arcs
/// added at the end ([`MinCostFlow::solve_from`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Basis {
    nodes: usize,
    arcs: usize,
    /// The state of each arc and then of each artificial arc.
    state: Vec<i8>,
    /// Whether every artificial arc leaves the root.
    from_root: bool,
}

/// A minimum-cost flow problem.
#[derive(Debug, Clone, Default)]
pub struct MinCostFlow {
    supply: Vec<i64>,
    tail: Vec<u32>,
    head: Vec<u32>,
    cost: Vec<i64>,
    cap: Vec<i64>,
    block: Option<usize>,
}

impl MinCostFlow {
    /// A problem on `nodes` nodes, numbered from 0, with no arcs and zero
    /// supplies.
    #[must_use]
    pub fn new(nodes: usize) -> Self {
        Self {
            supply: vec![0; nodes],
            ..Self::default()
        }
    }

    /// Reserve room for `arcs` arcs.
    pub fn reserve_arcs(&mut self, arcs: usize) {
        self.tail.reserve(arcs);
        self.head.reserve(arcs);
        self.cost.reserve(arcs);
        self.cap.reserve(arcs);
    }

    /// Set the number of arcs scanned in one block of the pricing search
    /// (at least 1). The default is the square root of the number of arcs, at
    /// least 10. A smaller block prices each pivot faster and looks at fewer
    /// arcs for the one to enter, which suits a problem whose pivots are cheap.
    /// The optimal cost does not depend on it, and the flow can differ only
    /// among equal-cost optima.
    pub fn set_pricing_block(&mut self, arcs: usize) {
        self.block = Some(arcs.max(1));
    }

    /// Number of nodes.
    #[must_use]
    pub fn nodes(&self) -> usize {
        self.supply.len()
    }

    /// Number of arcs.
    #[must_use]
    pub fn arcs(&self) -> usize {
        self.tail.len()
    }

    /// Set the supply of a node. A negative supply is a demand.
    ///
    /// # Panics
    ///
    /// When `node` is not a node of the problem.
    pub fn set_supply(&mut self, node: usize, supply: i64) {
        self.supply[node] = supply;
    }

    /// Add an arc from `from` to `to` with the capacity and cost given, and
    /// return its position in [`FlowSolution::flow`]. An arc with a node out of
    /// range or a negative capacity is reported by [`MinCostFlow::solve`].
    pub fn add_arc(&mut self, from: usize, to: usize, capacity: i64, cost: i64) -> usize {
        let bad = from >= self.supply.len() || to >= self.supply.len();
        let (from, to, capacity) = if bad {
            (NONE, NONE, -1)
        } else {
            (
                u32::try_from(from).unwrap_or(NONE),
                u32::try_from(to).unwrap_or(NONE),
                capacity,
            )
        };
        self.tail.push(from);
        self.head.push(to);
        self.cost.push(cost);
        self.cap.push(capacity);
        self.tail.len() - 1
    }

    /// Find a minimum-cost flow.
    ///
    /// # Errors
    ///
    /// [`FlowError::Infeasible`] when no flow meets the supplies,
    /// [`FlowError::Overflow`] when the data are too large, and
    /// [`FlowError::InvalidInput`] for an arc with an unknown node or a
    /// negative capacity.
    pub fn solve(&self) -> Result<FlowSolution, FlowError> {
        Solver::new(self)?.run()
    }

    /// Find a minimum-cost flow starting from the basis of an earlier solve.
    ///
    /// The problem must have the same nodes and supplies as the one that gave
    /// `basis`, and the arcs of that problem first, in the same order, with
    /// any new arcs after them. The new arcs start at their lower bound. The
    /// flow and the optimal cost are the same as [`MinCostFlow::solve`] gives,
    /// and the optimal flow can differ only among equal-cost optima. A basis
    /// that does not fit the problem is ignored and the solve starts from
    /// scratch.
    ///
    /// # Errors
    ///
    /// As [`MinCostFlow::solve`].
    pub fn solve_from(&self, basis: &Basis) -> Result<FlowSolution, FlowError> {
        let mut solver = Solver::new(self)?;
        solver.adopt(self, basis);
        solver.run()
    }

    /// Find a minimum-cost flow starting from a feasible flow, one value per
    /// arc in the order the arcs were added.
    ///
    /// The result is as [`MinCostFlow::solve`] gives it. A start that is not
    /// feasible, or whose arcs strictly between their bounds close a cycle, is
    /// ignored and the solve starts from scratch.
    ///
    /// # Errors
    ///
    /// As [`MinCostFlow::solve`].
    pub fn solve_from_flow(&self, flow: &[i64]) -> Result<FlowSolution, FlowError> {
        let mut solver = Solver::new(self)?;
        solver.adopt_flow(self, flow);
        solver.run()
    }
}

/// The representative of the set of `v` in a disjoint-set forest, with path
/// halving.
fn find(component: &mut [u32], mut v: u32) -> u32 {
    while component[v as usize] != v {
        component[v as usize] = component[component[v as usize] as usize];
        v = component[v as usize];
    }
    v
}

/// The working state of one solve. Arc `m + v` is the artificial arc of node
/// `v`, and node `n` is the root.
struct Solver {
    n: usize,
    m: usize,
    tail: Vec<u32>,
    head: Vec<u32>,
    cost: Vec<i64>,
    cap: Vec<i64>,
    flow: Vec<i64>,
    state: Vec<i8>,
    parent: Vec<u32>,
    pred: Vec<u32>,
    /// The arc to the parent leaves the node (it is directed from child to
    /// parent).
    up: Vec<bool>,
    depth: Vec<u32>,
    pi: Vec<i64>,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
    prev_sibling: Vec<u32>,
    next_arc: usize,
    block: usize,
    pivots: u64,
    stack: Vec<u32>,
    path: Vec<u32>,
    /// Whether every artificial arc leaves the root.
    from_root: bool,
}

impl Solver {
    fn new(problem: &MinCostFlow) -> Result<Self, FlowError> {
        let n = problem.supply.len();
        let m = problem.tail.len();
        if n >= NONE as usize || m >= NONE as usize - n {
            return Err(FlowError::Overflow);
        }
        if problem.cap.iter().any(|cap| *cap < 0) {
            return Err(FlowError::InvalidInput);
        }
        if problem.tail.contains(&NONE) {
            return Err(FlowError::InvalidInput);
        }
        let mut total: i128 = 0;
        let mut positive: i128 = 0;
        for supply in &problem.supply {
            total += i128::from(*supply);
            positive += i128::from((*supply).max(0));
        }
        if total != 0 {
            return Err(FlowError::Infeasible);
        }
        let max_cost = problem
            .cost
            .iter()
            .map(|cost| i128::from(*cost).abs())
            .max()
            .unwrap_or(0);
        let limit = i128::from(i64::MAX / 4);
        if positive > limit || 4 * (n as i128 + 2) * (max_cost + 1) > limit {
            return Err(FlowError::Overflow);
        }
        let inf = i64::MAX / 4;
        let big_m =
            i64::try_from((n as i128 + 1) * (max_cost + 1)).map_err(|_| FlowError::Overflow)?;
        let root = n as u32;

        let total_arcs = m + n;
        let mut tail = Vec::with_capacity(total_arcs);
        let mut head = Vec::with_capacity(total_arcs);
        let mut cost = Vec::with_capacity(total_arcs);
        let mut cap = Vec::with_capacity(total_arcs);
        let mut flow = vec![0_i64; total_arcs];
        let mut state = vec![LOWER; total_arcs];
        tail.extend_from_slice(&problem.tail);
        head.extend_from_slice(&problem.head);
        cost.extend_from_slice(&problem.cost);
        cap.extend_from_slice(&problem.cap);
        // An arc with no capacity cannot carry flow and is never priced.
        for arc in 0..m {
            if cap[arc] == 0 {
                state[arc] = TREE;
            }
        }

        let mut parent = vec![root; n + 1];
        let mut pred = vec![NONE; n + 1];
        let mut up = vec![false; n + 1];
        let mut depth = vec![1_u32; n + 1];
        let mut pi = vec![0_i64; n + 1];
        parent[n] = NONE;
        depth[n] = 0;
        for node in 0..n {
            let supply = problem.supply[node];
            let arc = m + node;
            pred[node] = arc as u32;
            cap.push(inf);
            cost.push(big_m);
            state[arc] = TREE;
            if supply >= 0 {
                // From the node to the root. With no supply it carries 0 and
                // points toward the root, which strong feasibility forbids, so
                // it points away instead.
                if supply > 0 {
                    tail.push(node as u32);
                    head.push(root);
                    up[node] = true;
                    pi[node] = -big_m;
                } else {
                    tail.push(root);
                    head.push(node as u32);
                    pi[node] = big_m;
                }
                flow[arc] = supply;
            } else {
                tail.push(root);
                head.push(node as u32);
                flow[arc] = -supply;
                pi[node] = big_m;
            }
        }

        let mut first_child = vec![NONE; n + 1];
        let mut next_sibling = vec![NONE; n + 1];
        let mut prev_sibling = vec![NONE; n + 1];
        // Children of the root in order of node number.
        for node in (0..n).rev() {
            let head_of_list = first_child[n];
            next_sibling[node] = head_of_list;
            if head_of_list != NONE {
                prev_sibling[head_of_list as usize] = node as u32;
            }
            first_child[n] = node as u32;
        }

        let total_arcs_f = total_arcs as f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the block size is the square root of an arc count, which fits"
        )]
        let block = problem
            .block
            .unwrap_or_else(|| (total_arcs_f.sqrt() as usize).max(10));
        Ok(Self {
            n,
            m,
            tail,
            head,
            cost,
            cap,
            flow,
            state,
            parent,
            pred,
            up,
            depth,
            pi,
            first_child,
            next_sibling,
            prev_sibling,
            next_arc: 0,
            block,
            pivots: 0,
            stack: Vec::new(),
            path: Vec::new(),
            from_root: false,
        })
    }

    /// Replace the starting tree by the one in `basis`, when it fits.
    /// Anything that does not fit leaves the start as it was.
    fn adopt(&mut self, problem: &MinCostFlow, basis: &Basis) {
        let (n, m) = (self.n, self.m);
        if basis.nodes != n || basis.arcs > m || basis.state.len() != basis.arcs + n {
            return;
        }
        let mut state = vec![LOWER; m + n];
        state[..basis.arcs].copy_from_slice(&basis.state[..basis.arcs]);
        state[m..].copy_from_slice(&basis.state[basis.arcs..]);
        if basis.from_root {
            self.install_from_root(problem, state);
        } else {
            self.install(problem, state);
        }
        // The arcs added since are the ones likely to price out, so the first
        // search starts at them.
        self.next_arc = basis.arcs;
    }

    /// [`Self::install`] with every artificial arc leaving the root. The
    /// arcs are left as they were when it fails.
    fn install_from_root(&mut self, problem: &MinCostFlow, state: Vec<i8>) {
        let (n, m) = (self.n, self.m);
        let root = n as u32;
        let saved: Vec<(u32, u32)> = (0..n)
            .map(|node| (self.tail[m + node], self.head[m + node]))
            .collect();
        for node in 0..n {
            self.tail[m + node] = root;
            self.head[m + node] = node as u32;
        }
        if self.install(problem, state) {
            self.from_root = true;
        } else {
            for (node, (t, h)) in saved.into_iter().enumerate() {
                self.tail[m + node] = t;
                self.head[m + node] = h;
            }
        }
    }

    /// Start from a feasible flow. Every arc strictly between its bounds is in
    /// the tree and must not close a cycle with the others. Each component of
    /// those arcs hangs from the root by an artificial arc that points away
    /// from the root and carries no flow, which keeps the tree strongly
    /// feasible. A flow that does not fit leaves the start as it was.
    fn adopt_flow(&mut self, problem: &MinCostFlow, flow: &[i64]) {
        let (n, m) = (self.n, self.m);
        if flow.len() != m {
            return;
        }
        let mut excess: Vec<i128> = problem.supply.iter().map(|s| i128::from(*s)).collect();
        for arc in 0..m {
            if flow[arc] < 0 || flow[arc] > self.cap[arc] {
                return;
            }
            excess[self.tail[arc] as usize] -= i128::from(flow[arc]);
            excess[self.head[arc] as usize] += i128::from(flow[arc]);
        }
        if excess.iter().any(|e| *e != 0) {
            return;
        }
        let mut component: Vec<u32> = (0..n as u32).collect();
        let mut state = vec![LOWER; m + n];
        for arc in 0..m {
            if self.cap[arc] == 0 {
                state[arc] = TREE;
            } else if flow[arc] == self.cap[arc] {
                state[arc] = UPPER;
            } else if flow[arc] > 0 {
                state[arc] = TREE;
                let t = find(&mut component, self.tail[arc]);
                let h = find(&mut component, self.head[arc]);
                if t == h {
                    return;
                }
                component[t as usize] = h;
            }
        }
        let mut seen = vec![false; n];
        for node in 0..n {
            let r = find(&mut component, node as u32) as usize;
            if !seen[r] {
                seen[r] = true;
                state[m + r] = TREE;
            }
        }
        // Every artificial arc leaves the root, with no flow.
        self.install_from_root(problem, state);
    }

    /// Build the tree, the flows, the potentials and the child lists from the
    /// arcs in the tree and the bound of every other arc. The flow of the tree
    /// arcs follows from the supplies. When the arcs do not form a spanning
    /// tree or the flow is not within the capacities, nothing changes.
    fn install(&mut self, problem: &MinCostFlow, state: Vec<i8>) -> bool {
        let (n, m) = (self.n, self.m);
        let is_tree =
            |arc: usize, state: &[i8]| state[arc] == TREE && (arc >= m || self.cap[arc] > 0);
        // Adjacency of the tree arcs, as lists of arc numbers per node.
        let mut degree = vec![0_u32; n + 2];
        let mut tree_arcs = 0;
        for arc in 0..m + n {
            if is_tree(arc, &state) {
                tree_arcs += 1;
                degree[self.tail[arc] as usize + 1] += 1;
                degree[self.head[arc] as usize + 1] += 1;
            }
        }
        if tree_arcs != n {
            return false;
        }
        for v in 0..=n {
            degree[v + 1] += degree[v];
        }
        let mut fill = degree.clone();
        let mut adjacent = vec![0_u32; 2 * n];
        for arc in 0..m + n {
            if is_tree(arc, &state) {
                for end in [self.tail[arc] as usize, self.head[arc] as usize] {
                    adjacent[fill[end] as usize] = arc as u32;
                    fill[end] += 1;
                }
            }
        }
        // Breadth-first from the root.
        let root = n;
        let mut parent = vec![NONE; n + 1];
        let mut pred = vec![NONE; n + 1];
        let mut up = vec![false; n + 1];
        let mut depth = vec![0_u32; n + 1];
        let mut pi = vec![0_i64; n + 1];
        let mut order: Vec<u32> = Vec::with_capacity(n + 1);
        let mut seen = vec![false; n + 1];
        seen[root] = true;
        order.push(root as u32);
        let mut at = 0;
        while at < order.len() {
            let v = order[at] as usize;
            at += 1;
            for slot in degree[v]..degree[v + 1] {
                let arc = adjacent[slot as usize] as usize;
                let (t, h) = (self.tail[arc] as usize, self.head[arc] as usize);
                let w = if t == v { h } else { t };
                if seen[w] {
                    continue;
                }
                seen[w] = true;
                parent[w] = v as u32;
                pred[w] = arc as u32;
                up[w] = t == w;
                depth[w] = depth[v] + 1;
                pi[w] = if up[w] {
                    pi[v] - self.cost[arc]
                } else {
                    pi[v] + self.cost[arc]
                };
                order.push(w as u32);
            }
        }
        if order.len() != n + 1 {
            return false;
        }
        // Flows: non-tree arcs at their bounds, tree arcs from the leaves up.
        let mut flow = vec![0_i64; m + n];
        let mut excess = vec![0_i128; n + 1];
        for (node, slot) in excess.iter_mut().enumerate().take(n) {
            *slot = i128::from(problem.supply[node]);
        }
        for arc in 0..m + n {
            if state[arc] == UPPER {
                flow[arc] = self.cap[arc];
                excess[self.tail[arc] as usize] -= i128::from(flow[arc]);
                excess[self.head[arc] as usize] += i128::from(flow[arc]);
            }
        }
        for &v in order.iter().skip(1).rev() {
            let v = v as usize;
            let arc = pred[v] as usize;
            let parent_of = parent[v] as usize;
            let x = if up[v] { excess[v] } else { -excess[v] };
            if x < 0 || x > i128::from(self.cap[arc]) {
                return false;
            }
            // In range of the arc's capacity, so it fits i64.
            flow[arc] = i64::try_from(x).unwrap_or(0);
            if up[v] {
                excess[parent_of] += x;
            } else {
                excess[parent_of] -= x;
            }
            excess[v] = 0;
        }
        let mut first_child = vec![NONE; n + 1];
        let mut next_sibling = vec![NONE; n + 1];
        let mut prev_sibling = vec![NONE; n + 1];
        for &v in order.iter().skip(1).rev() {
            let p = parent[v as usize] as usize;
            let head_of_list = first_child[p];
            next_sibling[v as usize] = head_of_list;
            if head_of_list != NONE {
                prev_sibling[head_of_list as usize] = v;
            }
            first_child[p] = v;
        }
        self.state = state;
        self.flow = flow;
        self.parent = parent;
        self.pred = pred;
        self.up = up;
        self.depth = depth;
        self.pi = pi;
        self.first_child = first_child;
        self.next_sibling = next_sibling;
        self.prev_sibling = prev_sibling;
        true
    }

    fn reduced_cost(&self, arc: usize) -> i64 {
        self.cost[arc] + self.pi[self.tail[arc] as usize] - self.pi[self.head[arc] as usize]
    }

    /// The arc to enter the tree by block search, or `None` at an optimum.
    fn find_entering(&mut self) -> Option<usize> {
        let total = self.m + self.n;
        let mut best: Option<usize> = None;
        let mut best_value = 0_i64;
        let mut count = 0;
        let mut arc = self.next_arc;
        for _ in 0..total {
            let state = self.state[arc];
            if state != TREE {
                let value = i64::from(state) * self.reduced_cost(arc);
                if value < best_value {
                    best_value = value;
                    best = Some(arc);
                }
            }
            count += 1;
            arc += 1;
            if arc == total {
                arc = 0;
            }
            if count == self.block {
                if best.is_some() {
                    break;
                }
                count = 0;
            }
        }
        if let Some(entering) = best {
            self.next_arc = arc;
            return Some(entering);
        }
        None
    }

    fn remove_child(&mut self, node: u32) {
        let parent = self.parent[node as usize] as usize;
        let previous = self.prev_sibling[node as usize];
        let next = self.next_sibling[node as usize];
        if previous == NONE {
            self.first_child[parent] = next;
        } else {
            self.next_sibling[previous as usize] = next;
        }
        if next != NONE {
            self.prev_sibling[next as usize] = previous;
        }
        self.prev_sibling[node as usize] = NONE;
        self.next_sibling[node as usize] = NONE;
    }

    fn add_child(&mut self, parent: u32, node: u32) {
        let first = self.first_child[parent as usize];
        self.next_sibling[node as usize] = first;
        self.prev_sibling[node as usize] = NONE;
        if first != NONE {
            self.prev_sibling[first as usize] = node;
        }
        self.first_child[parent as usize] = node;
        self.parent[node as usize] = parent;
    }

    /// Residual capacity of the tree arc above `node` when the cycle walks it
    /// downward (from the parent to the node).
    fn residual_down(&self, node: usize) -> i64 {
        let arc = self.pred[node] as usize;
        if self.up[node] {
            self.flow[arc]
        } else {
            self.cap[arc] - self.flow[arc]
        }
    }

    /// As [`Self::residual_down`], walking upward.
    fn residual_up(&self, node: usize) -> i64 {
        let arc = self.pred[node] as usize;
        if self.up[node] {
            self.cap[arc] - self.flow[arc]
        } else {
            self.flow[arc]
        }
    }

    fn pivot(&mut self, entering: usize) {
        self.pivots += 1;
        let at_lower = self.state[entering] == LOWER;
        let (tail, head) = (self.tail[entering] as usize, self.head[entering] as usize);
        let (first, second) = if at_lower { (tail, head) } else { (head, tail) };

        // The apex of the cycle.
        let (mut a, mut b) = (first, second);
        while a != b {
            match self.depth[a].cmp(&self.depth[b]) {
                std::cmp::Ordering::Greater => a = self.parent[a] as usize,
                std::cmp::Ordering::Less => b = self.parent[b] as usize,
                std::cmp::Ordering::Equal => {
                    a = self.parent[a] as usize;
                    b = self.parent[b] as usize;
                }
            }
        }
        let apex = a;

        let mut delta = self.cap[entering];
        let mut node = first;
        while node != apex {
            delta = delta.min(self.residual_down(node));
            node = self.parent[node] as usize;
        }
        node = second;
        while node != apex {
            delta = delta.min(self.residual_up(node));
            node = self.parent[node] as usize;
        }

        // Cunningham's rule: the last blocking arc met on the walk from the
        // apex along the flow change. That walk goes down to `first`, over the
        // entering arc, and up from `second` to the apex.
        let mut leaving_node = NONE;
        let mut on_first_side = false;
        node = second;
        while node != apex {
            if self.residual_up(node) == delta {
                leaving_node = node as u32;
            }
            node = self.parent[node] as usize;
        }
        if leaving_node == NONE && self.cap[entering] != delta {
            node = first;
            while node != apex {
                if self.residual_down(node) == delta {
                    leaving_node = node as u32;
                    on_first_side = true;
                    break;
                }
                node = self.parent[node] as usize;
            }
        }

        if delta > 0 {
            node = first;
            while node != apex {
                let arc = self.pred[node] as usize;
                if self.up[node] {
                    self.flow[arc] -= delta;
                } else {
                    self.flow[arc] += delta;
                }
                node = self.parent[node] as usize;
            }
            node = second;
            while node != apex {
                let arc = self.pred[node] as usize;
                if self.up[node] {
                    self.flow[arc] += delta;
                } else {
                    self.flow[arc] -= delta;
                }
                node = self.parent[node] as usize;
            }
            if at_lower {
                self.flow[entering] += delta;
            } else {
                self.flow[entering] -= delta;
            }
        }

        if leaving_node == NONE {
            // The entering arc is the blocking arc: it moves to its other
            // bound and the tree stays.
            self.state[entering] = if at_lower { UPPER } else { LOWER };
            return;
        }

        let leaving_arc = self.pred[leaving_node as usize] as usize;
        self.state[leaving_arc] = if self.flow[leaving_arc] == 0 {
            LOWER
        } else {
            UPPER
        };
        self.state[entering] = TREE;
        let (inside, outside) = if on_first_side {
            (first as u32, second as u32)
        } else {
            (second as u32, first as u32)
        };
        self.rehang(inside, outside, leaving_node, entering);
    }

    /// Hang the subtree below `leaving_node` from `outside` by the entering
    /// arc, which joins `outside` to `inside` in the subtree, and recompute the
    /// potentials and depths of the subtree.
    fn rehang(&mut self, inside: u32, outside: u32, leaving_node: u32, entering: usize) {
        // The path from `inside` up to the root of the subtree.
        self.path.clear();
        let mut node = inside;
        loop {
            self.path.push(node);
            if node == leaving_node {
                break;
            }
            node = self.parent[node as usize];
        }
        let path = std::mem::take(&mut self.path);
        let old_pred: Vec<u32> = path.iter().map(|v| self.pred[*v as usize]).collect();
        let old_up: Vec<bool> = path.iter().map(|v| self.up[*v as usize]).collect();
        for v in &path {
            self.remove_child(*v);
        }
        self.add_child(outside, inside);
        self.pred[inside as usize] = entering as u32;
        self.up[inside as usize] = self.tail[entering] == inside;
        for t in 0..path.len() - 1 {
            let child = path[t + 1];
            self.add_child(path[t], child);
            self.pred[child as usize] = old_pred[t];
            self.up[child as usize] = !old_up[t];
        }
        self.path = path;

        // Potentials and depths below `inside`, by a walk of the subtree.
        self.stack.clear();
        self.stack.push(inside);
        while let Some(v) = self.stack.pop() {
            let v = v as usize;
            let parent = self.parent[v] as usize;
            let arc = self.pred[v] as usize;
            self.pi[v] = if self.up[v] {
                self.pi[parent] - self.cost[arc]
            } else {
                self.pi[parent] + self.cost[arc]
            };
            self.depth[v] = self.depth[parent] + 1;
            let mut child = self.first_child[v];
            while child != NONE {
                self.stack.push(child);
                child = self.next_sibling[child as usize];
            }
        }
    }

    fn run(mut self) -> Result<FlowSolution, FlowError> {
        while let Some(entering) = self.find_entering() {
            self.pivot(entering);
        }
        if self.flow[self.m..].iter().any(|flow| *flow != 0) {
            return Err(FlowError::Infeasible);
        }
        let mut cost: i128 = 0;
        for arc in 0..self.m {
            cost += i128::from(self.cost[arc]) * i128::from(self.flow[arc]);
        }
        let basis = Basis {
            nodes: self.n,
            arcs: self.m,
            state: self.state,
            from_root: self.from_root,
        };
        self.flow.truncate(self.m);
        self.pi.truncate(self.n);
        Ok(FlowSolution {
            flow: self.flow,
            cost,
            potentials: self.pi,
            pivots: self.pivots,
            basis,
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "test sizes are tiny"
)]
mod tests {
    use super::*;

    fn solve(problem: &MinCostFlow) -> FlowSolution {
        problem.solve().expect("feasible")
    }

    /// Check flow conservation, capacities and the optimality conditions of
    /// the potentials.
    fn assert_certificate(problem: &MinCostFlow, solution: &FlowSolution) {
        let mut balance = problem.supply.clone();
        for arc in 0..problem.arcs() {
            let flow = solution.flow[arc];
            assert!(
                flow >= 0 && flow <= problem.cap[arc],
                "capacity of arc {arc}"
            );
            balance[problem.tail[arc] as usize] -= flow;
            balance[problem.head[arc] as usize] += flow;
            let reduced = problem.cost[arc] + solution.potentials[problem.tail[arc] as usize]
                - solution.potentials[problem.head[arc] as usize];
            if flow == 0 && problem.cap[arc] > 0 {
                assert!(reduced >= 0, "arc {arc} at 0 with reduced cost {reduced}");
            }
            if flow == problem.cap[arc] && flow > 0 {
                assert!(
                    reduced <= 0,
                    "arc {arc} at capacity with reduced cost {reduced}"
                );
            }
            if flow > 0 && flow < problem.cap[arc] {
                assert_eq!(reduced, 0, "arc {arc} strictly between its bounds");
            }
        }
        assert!(balance.iter().all(|b| *b == 0), "flow conservation");
    }

    #[test]
    fn two_nodes_one_arc() {
        let mut problem = MinCostFlow::new(2);
        problem.set_supply(0, 3);
        problem.set_supply(1, -3);
        problem.add_arc(0, 1, 5, 7);
        let solution = solve(&problem);
        assert_eq!(solution.flow, vec![3]);
        assert_eq!(solution.cost, 21);
        assert_certificate(&problem, &solution);
    }

    #[test]
    fn the_cheaper_route_fills_before_the_dearer_one() {
        // 0 -> 3 by node 1 (cost 1 + 1, capacity 2) or by node 2 (cost 2 + 3,
        // capacity 5). Four units: two cheap at 2 and two dear at 5.
        let mut problem = MinCostFlow::new(4);
        problem.set_supply(0, 4);
        problem.set_supply(3, -4);
        problem.add_arc(0, 1, 2, 1);
        problem.add_arc(1, 3, 2, 1);
        problem.add_arc(0, 2, 5, 2);
        problem.add_arc(2, 3, 5, 3);
        let solution = solve(&problem);
        assert_eq!(solution.flow, vec![2, 2, 2, 2]);
        assert_eq!(solution.cost, 14);
        assert_certificate(&problem, &solution);
    }

    #[test]
    fn a_two_by_two_assignment_by_hand() {
        // Costs [[4, 1], [2, 3]]: the diagonal costs 7 and the cross costs 3.
        let mut problem = MinCostFlow::new(4);
        problem.set_supply(0, 1);
        problem.set_supply(1, 1);
        problem.set_supply(2, -1);
        problem.set_supply(3, -1);
        problem.add_arc(0, 2, 1, 4);
        problem.add_arc(0, 3, 1, 1);
        problem.add_arc(1, 2, 1, 2);
        problem.add_arc(1, 3, 1, 3);
        let solution = solve(&problem);
        assert_eq!(solution.flow, vec![0, 1, 1, 0]);
        assert_eq!(solution.cost, 3);
        assert_certificate(&problem, &solution);
    }

    #[test]
    fn a_negative_cost_cycle_is_used_to_capacity() {
        // A circulation through 0 -> 1 -> 2 -> 0 with total cost -1 per unit
        // and capacity 4 on the narrowest arc.
        let mut problem = MinCostFlow::new(3);
        problem.add_arc(0, 1, 6, -2);
        problem.add_arc(1, 2, 4, 0);
        problem.add_arc(2, 0, 9, 1);
        let solution = solve(&problem);
        assert_eq!(solution.flow, vec![4, 4, 4]);
        assert_eq!(solution.cost, -4);
        assert_certificate(&problem, &solution);
    }

    #[test]
    fn a_transportation_problem_by_hand() {
        // Two warehouses (20 and 30) and three shops (10, 25, 15).
        let cost = [[8, 6, 10], [9, 12, 13]];
        let mut problem = MinCostFlow::new(5);
        problem.set_supply(0, 20);
        problem.set_supply(1, 30);
        problem.set_supply(2, -10);
        problem.set_supply(3, -25);
        problem.set_supply(4, -15);
        for (i, row) in cost.iter().enumerate() {
            for (j, c) in row.iter().enumerate() {
                problem.add_arc(i, 2 + j, 100, *c);
            }
        }
        let solution = solve(&problem);
        // Shop 2 (cost 6 from 0, 12 from 1) takes 20 from warehouse 0, the
        // rest of shop 2 and all of shops 1 and 3 come from warehouse 1.
        // 20*6 + 10*9 + 5*12 + 15*13 = 120 + 90 + 60 + 195 = 465.
        assert_eq!(solution.cost, 465);
        assert_certificate(&problem, &solution);
    }

    #[test]
    fn more_supply_than_capacity_is_infeasible() {
        let mut problem = MinCostFlow::new(2);
        problem.set_supply(0, 3);
        problem.set_supply(1, -3);
        problem.add_arc(0, 1, 2, 1);
        assert_eq!(problem.solve(), Err(FlowError::Infeasible));
    }

    #[test]
    fn supplies_that_do_not_sum_to_zero_are_infeasible() {
        let mut problem = MinCostFlow::new(2);
        problem.set_supply(0, 3);
        problem.set_supply(1, -2);
        problem.add_arc(0, 1, 5, 1);
        assert_eq!(problem.solve(), Err(FlowError::Infeasible));
    }

    #[test]
    fn invalid_arcs_are_reported() {
        let mut problem = MinCostFlow::new(2);
        problem.add_arc(0, 2, 1, 1);
        assert_eq!(problem.solve(), Err(FlowError::InvalidInput));
        let mut problem = MinCostFlow::new(2);
        problem.add_arc(0, 1, -1, 1);
        assert_eq!(problem.solve(), Err(FlowError::InvalidInput));
    }

    #[test]
    fn huge_costs_are_refused() {
        let mut problem = MinCostFlow::new(2);
        problem.set_supply(0, 1);
        problem.set_supply(1, -1);
        problem.add_arc(0, 1, 1, i64::MAX / 2);
        assert_eq!(problem.solve(), Err(FlowError::Overflow));
    }

    #[test]
    fn an_empty_problem_has_the_empty_flow() {
        let solution = solve(&MinCostFlow::new(3));
        assert_eq!(solution.flow.len(), 0);
        assert_eq!(solution.cost, 0);
    }

    #[test]
    fn the_same_problem_gives_the_same_flow() {
        let mut state = 7_u64;
        let problem = random_network(&mut state, 12, 40);
        assert_eq!(problem.solve(), problem.solve());
    }

    /// SplitMix64, which is enough to draw test problems.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(state: &mut u64, bound: u64) -> u64 {
        next(state) % bound
    }

    fn random_network(state: &mut u64, nodes: usize, arcs: usize) -> MinCostFlow {
        let mut problem = MinCostFlow::new(nodes);
        for _ in 0..arcs {
            let from = below(state, nodes as u64) as usize;
            let to = below(state, nodes as u64) as usize;
            let capacity = below(state, 6) as i64;
            let cost = below(state, 21) as i64 - 6;
            problem.add_arc(from, to, capacity, cost);
        }
        // Supplies from a random feasible flow, so the problem is feasible.
        for _ in 0..nodes {
            let from = below(state, nodes as u64) as usize;
            let to = below(state, nodes as u64) as usize;
            let amount = below(state, 3) as i64;
            problem.supply[from] += amount;
            problem.supply[to] -= amount;
        }
        problem
    }

    /// Minimum cost by successive shortest paths with Bellman-Ford, after
    /// saturating the negative-cost arcs, as an independent reference. Returns
    /// `None` when infeasible.
    fn reference_cost(problem: &MinCostFlow) -> Option<i128> {
        let n = problem.nodes();
        let super_source = n;
        let super_sink = n + 1;
        // Residual graph as arrays of (to, cap, cost) with paired reverse arcs.
        let mut to: Vec<usize> = Vec::new();
        let mut cap: Vec<i64> = Vec::new();
        let mut cost: Vec<i64> = Vec::new();
        let mut from: Vec<usize> = Vec::new();
        let mut add = |u: usize, v: usize, c: i64, w: i64| {
            from.push(u);
            to.push(v);
            cap.push(c);
            cost.push(w);
            from.push(v);
            to.push(u);
            cap.push(0);
            cost.push(-w);
        };
        let mut excess = problem.supply.clone();
        let mut constant: i128 = 0;
        for arc in 0..problem.arcs() {
            let (u, v) = (problem.tail[arc] as usize, problem.head[arc] as usize);
            let (c, w) = (problem.cap[arc], problem.cost[arc]);
            if w < 0 {
                // Saturate, and offer the reverse arc at the positive cost.
                constant += i128::from(c) * i128::from(w);
                excess[u] -= c;
                excess[v] += c;
                add(v, u, c, -w);
            } else {
                add(u, v, c, w);
            }
        }
        let mut required = 0;
        for (node, e) in excess.iter().enumerate() {
            if *e > 0 {
                add(super_source, node, *e, 0);
                required += *e;
            } else if *e < 0 {
                add(node, super_sink, -*e, 0);
            }
        }
        let total_nodes = n + 2;
        let mut sent = 0;
        let mut total: i128 = 0;
        while sent < required {
            let mut dist = vec![i64::MAX; total_nodes];
            let mut via = vec![usize::MAX; total_nodes];
            dist[super_source] = 0;
            for _ in 0..total_nodes {
                let mut changed = false;
                for a in 0..to.len() {
                    if cap[a] > 0
                        && dist[from[a]] != i64::MAX
                        && dist[from[a]] + cost[a] < dist[to[a]]
                    {
                        dist[to[a]] = dist[from[a]] + cost[a];
                        via[to[a]] = a;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            if dist[super_sink] == i64::MAX {
                return None;
            }
            let mut push = required - sent;
            let mut node = super_sink;
            while node != super_source {
                push = push.min(cap[via[node]]);
                node = from[via[node]];
            }
            node = super_sink;
            while node != super_source {
                cap[via[node]] -= push;
                cap[via[node] ^ 1] += push;
                node = from[via[node]];
            }
            sent += push;
            total += i128::from(push) * i128::from(dist[super_sink]);
        }
        Some(total + constant)
    }

    #[test]
    fn random_networks_agree_with_successive_shortest_paths() {
        let mut state = 20_261_121_u64;
        let mut feasible = 0;
        for round in 0..400 {
            let nodes = 3 + below(&mut state, 8) as usize;
            let arcs = 4 + below(&mut state, 30) as usize;
            let problem = random_network(&mut state, nodes, arcs);
            let expected = reference_cost(&problem);
            match problem.solve() {
                Ok(solution) => {
                    feasible += 1;
                    assert_eq!(Some(solution.cost), expected, "round {round}");
                    assert_certificate(&problem, &solution);
                }
                Err(FlowError::Infeasible) => assert_eq!(expected, None, "round {round}"),
                Err(other) => panic!("round {round}: {other}"),
            }
        }
        assert!(feasible > 100, "only {feasible} feasible problems drawn");
    }

    /// A copy of `problem` with other costs on the same arcs.
    fn recosted(problem: &MinCostFlow, state: &mut u64) -> MinCostFlow {
        let mut other = problem.clone();
        for cost in &mut other.cost {
            *cost = below(state, 21) as i64 - 6;
        }
        other
    }

    #[test]
    fn warm_starts_give_the_optimal_cost() {
        let mut state = 77_u64;
        let mut checked = 0;
        for round in 0..600 {
            let nodes = 3 + below(&mut state, 9) as usize;
            let arcs = 4 + below(&mut state, 30) as usize;
            let problem = random_network(&mut state, nodes, arcs);
            let Ok(first) = problem.solve() else { continue };
            checked += 1;
            let expected = Some(first.cost);

            // The same problem from its own flow and basis.
            let again = problem.solve_from_flow(&first.flow).expect("feasible");
            assert_eq!(Some(again.cost), expected, "round {round}: own flow");
            assert_certificate(&problem, &again);
            let again = problem.solve_from(&first.basis).expect("feasible");
            assert_eq!(Some(again.cost), expected, "round {round}: own basis");

            // Other costs on the same arcs: the flow stays feasible.
            let other = recosted(&problem, &mut state);
            let from_flow = other.solve_from_flow(&first.flow).expect("feasible");
            assert_eq!(
                Some(from_flow.cost),
                reference_cost(&other),
                "round {round}: flow of other costs"
            );
            assert_certificate(&other, &from_flow);
            let from_basis = other.solve_from(&first.basis).expect("feasible");
            assert_eq!(Some(from_basis.cost), Some(from_flow.cost), "round {round}");

            // More arcs, from the basis of the smaller problem and from that of
            // a solve started at a flow.
            let mut larger = problem.clone();
            for _ in 0..=below(&mut state, 8) {
                let from = below(&mut state, nodes as u64) as usize;
                let to = below(&mut state, nodes as u64) as usize;
                larger.add_arc(
                    from,
                    to,
                    below(&mut state, 6) as i64,
                    below(&mut state, 21) as i64 - 6,
                );
            }
            let cold = larger.solve().expect("feasible");
            for basis in [&first.basis, &again.basis, &from_flow.basis] {
                if basis == &from_flow.basis && other.arcs() != problem.arcs() {
                    continue;
                }
                let warm = larger.solve_from(basis).expect("feasible");
                assert_eq!(warm.cost, cold.cost, "round {round}: more arcs");
                assert_certificate(&larger, &warm);
            }
        }
        assert!(checked > 150, "only {checked} feasible problems drawn");
    }

    #[test]
    fn a_start_that_does_not_fit_is_ignored() {
        let mut problem = MinCostFlow::new(3);
        problem.set_supply(0, 2);
        problem.set_supply(2, -2);
        problem.add_arc(0, 1, 2, 1);
        problem.add_arc(1, 2, 2, 1);
        problem.add_arc(0, 2, 2, 5);
        let cold = problem.solve().expect("feasible");
        // Wrong length, over capacity and not conserving flow.
        for bad in [vec![1], vec![3, 3, 0], vec![1, 0, 0]] {
            let solution = problem.solve_from_flow(&bad).expect("feasible");
            assert_eq!(solution.cost, cold.cost);
        }
        // A basis from a problem with other nodes.
        let mut other = MinCostFlow::new(2);
        other.set_supply(0, 1);
        other.set_supply(1, -1);
        other.add_arc(0, 1, 1, 1);
        let basis = other.solve().expect("feasible").basis;
        assert_eq!(
            problem.solve_from(&basis).expect("feasible").cost,
            cold.cost
        );
    }

    #[test]
    fn the_pricing_block_does_not_change_the_cost() {
        let mut state = 5_u64;
        for round in 0..200 {
            let nodes = 3 + below(&mut state, 9) as usize;
            let arcs = 4 + below(&mut state, 30) as usize;
            let mut problem = random_network(&mut state, nodes, arcs);
            let Ok(default) = problem.solve() else {
                continue;
            };
            for block in [1, 2, 7] {
                problem.set_pricing_block(block);
                let solution = problem.solve().expect("feasible");
                assert_eq!(solution.cost, default.cost, "round {round} block {block}");
                assert_certificate(&problem, &solution);
            }
        }
    }

    /// The cheapest assignment of `rows` rows to distinct columns, by trying
    /// every injection, with `None` for a forbidden pair.
    fn brute_force_assignment(costs: &[Vec<Option<i64>>]) -> Option<i64> {
        fn go(row: usize, costs: &[Vec<Option<i64>>], used: &mut Vec<bool>) -> Option<i64> {
            if row == costs.len() {
                return Some(0);
            }
            let mut best: Option<i64> = None;
            for col in 0..used.len() {
                if used[col] {
                    continue;
                }
                if let Some(c) = costs[row][col] {
                    used[col] = true;
                    if let Some(rest) = go(row + 1, costs, used) {
                        best = Some(best.map_or(c + rest, |b| b.min(c + rest)));
                    }
                    used[col] = false;
                }
            }
            best
        }
        let cols = costs.first().map_or(0, Vec::len);
        go(0, costs, &mut vec![false; cols])
    }

    #[test]
    fn random_assignments_agree_with_brute_force() {
        let mut state = 99_u64;
        for round in 0..300 {
            let rows = 1 + below(&mut state, 5) as usize;
            let cols = rows + below(&mut state, 3) as usize;
            let costs: Vec<Vec<Option<i64>>> = (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| (below(&mut state, 4) != 0).then(|| below(&mut state, 30) as i64))
                        .collect()
                })
                .collect();
            let mut problem = MinCostFlow::new(rows + cols);
            for r in 0..rows {
                problem.set_supply(r, 1);
            }
            // Every column takes at most one row; the unused ones are fed by
            // a free arc from a slack node.
            let slack_node = problem.supply.len();
            problem.supply.push(0);
            problem.set_supply(slack_node, -(rows as i64));
            let mut sink_problem = MinCostFlow::new(rows + cols + 1);
            sink_problem.supply = problem.supply.clone();
            for (r, row) in costs.iter().enumerate() {
                for (c, cost) in row.iter().enumerate() {
                    if let Some(cost) = cost {
                        sink_problem.add_arc(r, rows + c, 1, *cost);
                    }
                }
            }
            for c in 0..cols {
                sink_problem.add_arc(rows + c, slack_node, 1, 0);
            }
            let expected = brute_force_assignment(&costs);
            match sink_problem.solve() {
                Ok(solution) => {
                    assert_eq!(
                        Some(solution.cost),
                        expected.map(i128::from),
                        "round {round}"
                    );
                    assert_certificate(&sink_problem, &solution);
                }
                Err(FlowError::Infeasible) => assert_eq!(expected, None, "round {round}"),
                Err(other) => panic!("round {round}: {other}"),
            }
        }
    }

    #[test]
    fn a_degenerate_assignment_with_equal_costs_terminates() {
        // Every cost equal: a worst case for stalling on degenerate pivots.
        let k = 30;
        let mut problem = MinCostFlow::new(2 * k);
        for i in 0..k {
            problem.set_supply(i, 1);
            problem.set_supply(k + i, -1);
            for j in 0..k {
                problem.add_arc(i, k + j, 1, 5);
            }
        }
        let solution = solve(&problem);
        assert_eq!(solution.cost, 5 * k as i128);
        assert_certificate(&problem, &solution);
    }
}
