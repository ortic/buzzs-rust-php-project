// buzzs-rust-php-project - Copyright (c) 2025-2026 David Buzz <davidbuzz@gmail.com>
// Licensed under the BSD-3-Clause, the license PHP master uses; see LICENSE.
//! The candidate subgraph and its analysis: php-src's `gc_mark_roots`
//! (trial deletion), `gc_scan_roots` (scan black/white) and
//! `gc_collect_roots` (collect white), restated over `Rc` counts.
//!
//! Every node is held by exactly one handle owned by the graph, so
//! `strong_count - 1` is the number of references the rest of the program
//! holds. Edges are recorded with multiplicity while each node's contents
//! are enumerated (each owned handle once), giving `internal`: the
//! references coming from candidate nodes. `external = strong - 1 -
//! internal > 0` makes a node live; liveness then propagates along edges.

use super::traverse::{lookup, GcVisitor};
use crate::object::{self, ObjectData};
use php_runtime::{Obj, ObjWeak, PhpRef, Value, ZArray};
use rustc_hash::FxHashMap;

/// No node.
pub(super) const NONE: u32 = u32::MAX;

/// The graph's handle on a node (the one reference the collector owns).
pub(super) enum Handle {
    /// An object.
    Obj(Obj),
    /// An array table.
    Arr(ZArray),
    /// A PHP reference.
    Ref(PhpRef),
    /// Placeholder while the node is being expanded / after it was moved
    /// out.
    Taken,
}

/// A candidate node: its handle and its children. What every edge into
/// it reads or writes (the count of such edges, the marks) is kept beside
/// the node list, in [`Graph::counts`] and [`Graph::flags`]: a few
/// bytes a node, in one block each, where the nodes are 32 bytes each in
/// chunks. Those are the random accesses of a run (one per edge while the
/// graph is built, one per edge while liveness spreads).
pub(super) struct Node {
    pub handle: Handle,
    /// Children: `edges[start..start + len]`.
    start: u32,
    len: u32,
}

/// A node's reference counts ([`Graph::counts`]).
#[derive(Clone, Copy)]
struct Counts {
    /// The references from candidate nodes (edges into the node).
    internal: u32,
    /// The handles other than the graph's own, read when the node is
    /// expanded (its allocation is at hand then; the counts do not move
    /// while the graph is built: nothing runs, and the graph takes its
    /// one handle of a node before it expands it). The scan then decides
    /// liveness from this array and the marks alone, without a second
    /// walk over every object, array and reference of the heap.
    others: u32,
}

/// [`Graph::flags`]: for objects, an edge to the object was seen (objects
/// found only through the object store were not; every reference to them
/// is external).
const REACHED: u8 = 1;
/// [`Graph::flags`]: must be considered live whatever the counts say
/// (contents could not be enumerated).
const PINNED: u8 = 2;
/// [`Graph::flags`]: scanned black.
pub(super) const LIVE: u8 = 4;

impl Node {
    /// Whether php-src counts this node in `gc_collect_cycles()`'s result
    /// (objects and arrays; "don't count references for compatibility").
    pub fn counted(&self) -> bool {
        !matches!(self.handle, Handle::Ref(_))
    }

    /// The object data, if this is an object.
    pub fn data(&self) -> Option<&ObjectData> {
        match &self.handle {
            Handle::Obj(o) => Some(object::data(o)),
            _ => None,
        }
    }
}

/// A `WeakMap` entry: `value` is reachable only while both the map and
/// the key are (`GC_FROM_WEAKMAP` / `GC_FROM_WEAKMAP_KEY`).
pub(super) struct CondEdge {
    pub map: u32,
    /// The key object's handle, if it is alive (resolved to a node by
    /// `analyze`).
    key_handle: Option<u32>,
    pub key: u32,
    pub value: u32,
}

/// Nodes per [`Nodes`] chunk.
const CHUNK: usize = 4096;

/// The node list, in fixed-size chunks: a graph spanning a large heap
/// grows without the reallocation copies (and the doubled capacity) of a
/// single `Vec`, which would raise the collector's peak memory by up to
/// three times the graph's size (bug77345_gc_2 under `memory_limit`).
#[derive(Default)]
pub(super) struct Nodes {
    chunks: Vec<Vec<Node>>,
    len: usize,
}

impl Nodes {
    pub fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, n: Node) {
        match self.chunks.last_mut() {
            Some(c) if c.len() < CHUNK => c.push(n),
            _ => {
                // The first chunk grows like a `Vec` (small graphs stay
                // small); later ones are allocated whole.
                let mut c = if self.chunks.is_empty() {
                    Vec::new()
                } else {
                    Vec::with_capacity(CHUNK)
                };
                c.push(n);
                self.chunks.push(c);
            }
        }
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.chunks.iter().flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.chunks.iter_mut().flatten()
    }
}

impl std::ops::Index<usize> for Nodes {
    type Output = Node;

    fn index(&self, i: usize) -> &Node {
        &self.chunks[i / CHUNK][i % CHUNK]
    }
}

impl std::ops::IndexMut<usize> for Nodes {
    fn index_mut(&mut self, i: usize) -> &mut Node {
        &mut self.chunks[i / CHUNK][i % CHUNK]
    }
}

/// The candidate subgraph.
#[derive(Default)]
pub(super) struct Graph {
    pub nodes: Nodes,
    /// Per node: `REACHED`, `PINNED`, `LIVE`.
    flags: Vec<u8>,
    /// Per node, its two counts side by side (one entry to add, one line
    /// of memory to read in the scan).
    counts: Vec<Counts>,
    /// Object nodes by object handle (`NONE`: not in the graph). Handles
    /// are dense and cannot be reused while the graph holds the objects;
    /// this costs 4 bytes per handle instead of a hash entry per node.
    obj_index: Vec<u32>,
    /// Array and reference nodes by address (one lookup per edge to an
    /// array or a reference: under `SipHash` it was 4% of a run).
    index: FxHashMap<usize, u32>,
    edges: Vec<u32>,
    pub cond: Vec<CondEdge>,
    /// Nodes before this index have been expanded.
    expanded: usize,
}

impl Graph {
    /// Adds a candidate root.
    pub fn add_root(&mut self, v: &Value) {
        self.intern(v);
    }

    /// Adds a live object from the object store as a candidate root.
    pub fn add_object(&mut self, o: Obj) {
        if self.obj_node(object::data(&o).handle.get()).is_some() {
            return;
        }
        let i = self.push_node(Handle::Obj(o), false);
        self.set_obj_node(i);
    }

    /// The node of the object with handle `h` (unique among live objects).
    fn obj_node(&self, h: u32) -> Option<u32> {
        obj_node_in(&self.obj_index, h)
    }

    /// Indexes object node `i` by its handle.
    fn set_obj_node(&mut self, i: u32) {
        let Some(h) = self.nodes[i as usize]
            .data()
            .map(|d| d.handle.get() as usize)
        else {
            return;
        };
        if self.obj_index.len() <= h {
            self.obj_index.resize(h + 1, NONE);
        }
        self.obj_index[h] = i;
    }

    fn push_node(&mut self, handle: Handle, reached: bool) -> u32 {
        let i = u32::try_from(self.nodes.len()).unwrap_or(NONE - 1);
        self.nodes.push(Node {
            handle,
            start: 0,
            len: 0,
        });
        self.flags.push(if reached { REACHED } else { 0 });
        self.counts.push(Counts {
            internal: 0,
            others: u32::MAX,
        });
        i
    }

    fn push(&mut self, key: usize, handle: Handle) -> u32 {
        let i = self.push_node(handle, true);
        self.index.insert(key, i);
        i
    }

    /// The node for a collectable value (creating it), or `None` for
    /// scalars, strings and resources.
    fn intern(&mut self, v: &Value) -> Option<u32> {
        match v {
            Value::Object(o) => self.intern_obj(o),
            Value::Array(a) => Some(self.intern_arr(a)),
            Value::Reference(r) => Some(self.intern_ref(r)),
            _ => None,
        }
    }

    pub(super) fn intern_obj(&mut self, o: &Obj) -> Option<u32> {
        // Objects of other `PhpObject` implementations hold nothing the
        // collector knows about: leaves.
        let d = o.downcast_ref::<ObjectData>()?;
        if let Some(i) = self.obj_node(d.handle.get()) {
            if let Some(f) = self.flags.get_mut(i as usize) {
                *f |= REACHED;
            }
            return Some(i);
        }
        let i = self.push_node(Handle::Obj(o.clone()), true);
        self.set_obj_node(i);
        Some(i)
    }

    /// The node of an array table. A table with one handle (the array a
    /// property or an element holds for itself: most of a heap's arrays)
    /// is reached through that handle and no other, so it is neither
    /// looked up nor indexed: the node is made at once. (The graph's own
    /// handle is the second; nothing else makes one while it is built.)
    pub(super) fn intern_arr(&mut self, a: &ZArray) -> u32 {
        if a.refcount() == 1 {
            return self.push_node(Handle::Arr(a.clone()), true);
        }
        match self.index.get(&a.addr()) {
            Some(&i) => i,
            None => self.push(a.addr(), Handle::Arr(a.clone())),
        }
    }

    /// The node of a reference; one with a single handle is not indexed
    /// ([`Graph::intern_arr`]).
    pub(super) fn intern_ref(&mut self, r: &PhpRef) -> u32 {
        if r.refcount() == 1 {
            return self.push_node(Handle::Ref(r.clone()), true);
        }
        match self.index.get(&r.addr()) {
            Some(&i) => i,
            None => self.push(r.addr(), Handle::Ref(r.clone())),
        }
    }

    /// Records an edge from the node being expanded to `v`.
    pub(super) fn edge(&mut self, v: &Value) {
        if let Some(t) = self.intern(v) {
            self.link(t);
        }
    }

    pub(super) fn link(&mut self, t: u32) {
        self.edges.push(t);
        self.count_edge(t);
    }

    /// One more edge into node `t`.
    #[inline]
    fn count_edge(&mut self, t: u32) {
        if let Some(n) = self.counts.get_mut(t as usize) {
            n.internal += 1;
        }
    }

    /// Whether node `i` was scanned black (after [`Graph::analyze`]).
    #[inline]
    pub fn live(&self, i: usize) -> bool {
        self.flags.get(i).is_some_and(|f| f & LIVE != 0)
    }

    /// Hands the marks over (the free phase reads them while it empties
    /// the node list); every node then reads as not live.
    pub fn take_marks(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.flags)
    }

    /// The object handle of node `i`, if it is an object reached through
    /// an edge.
    pub fn obj(&self, i: usize) -> Option<&Obj> {
        match &self.nodes[i].handle {
            Handle::Obj(o) if self.flags.get(i).is_some_and(|f| f & REACHED != 0) => Some(o),
            _ => None,
        }
    }

    /// Records a `WeakMap` entry of the map being expanded.
    pub(super) fn cond_edge(&mut self, map: u32, key: &ObjWeak, v: &Value) {
        if let Some(t) = self.intern(v) {
            self.count_edge(t);
            self.cond.push(CondEdge {
                map,
                key_handle: key.upgrade().map(|o| object::data(&o).handle.get()),
                key: NONE,
                value: t,
            });
        }
    }

    /// Marks the node being expanded as live.
    pub(super) fn pin(&mut self, i: u32) {
        if let Some(f) = self.flags.get_mut(i as usize) {
            *f |= PINNED;
        }
    }

    /// Enumerates the contents of every node reachable from the roots.
    pub fn build(&mut self) {
        while self.expanded < self.nodes.len() {
            let i = self.expanded;
            self.expanded += 1;
            self.expand(i);
        }
    }

    fn expand(&mut self, i: usize) {
        let handle = std::mem::replace(&mut self.nodes[i].handle, Handle::Taken);
        let start = self.edges.len();
        let me = u32::try_from(i).unwrap_or(NONE - 1);
        let strong = match &handle {
            Handle::Obj(o) => o.refcount(),
            Handle::Arr(a) => a.refcount(),
            Handle::Ref(r) => r.refcount(),
            Handle::Taken => 0,
        };
        if let Some(n) = self.counts.get_mut(i) {
            // The graph holds one handle. (A count that does not fit
            // reads as a node with outside references: live.)
            n.others = strong
                .checked_sub(1)
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or(u32::MAX);
        }
        match &handle {
            Handle::Obj(o) => self.expand_object(me, object::data(o)),
            Handle::Arr(a) => {
                for v in a.values() {
                    self.edge(v);
                }
            }
            Handle::Ref(r) => match r.try_borrow() {
                Some(v) => self.edge(&v),
                None => self.pin(me),
            },
            Handle::Taken => {}
        }
        let n = &mut self.nodes[i];
        n.handle = handle;
        n.start = u32::try_from(start).unwrap_or(0);
        n.len = u32::try_from(self.edges.len() - start).unwrap_or(0);
    }

    fn expand_object(&mut self, me: u32, d: &ObjectData) {
        match d.slots.try_borrow() {
            Ok(slots) => {
                for v in slots.iter().flatten() {
                    self.edge(v);
                }
            }
            Err(_) => self.pin(me),
        }
        match d.dynamic.try_borrow() {
            Ok(dynamic) => {
                if let Some(t) = dynamic.as_ref() {
                    for v in t.values() {
                        self.edge(v);
                    }
                }
            }
            Err(_) => self.pin(me),
        }
        match d.internal.try_borrow() {
            Ok(state) => {
                if let Some(state) = state.as_ref() {
                    let any: &dyn std::any::Any = &**state;
                    if let Some(f) = lookup(any.type_id()) {
                        let mut v = GcVisitor::new(self, me);
                        f(any, &mut v);
                    }
                }
            }
            // Native code is working on the state right now.
            Err(_) => self.pin(me),
        }
    }

    /// Scan: decides which nodes are live. Returns the number of garbage
    /// nodes.
    pub fn analyze(&mut self) -> usize {
        for c in &mut self.cond {
            c.key = c
                .key_handle
                .and_then(|h| obj_node_in(&self.obj_index, h))
                .unwrap_or(NONE);
        }
        let mut work: Vec<u32> = Vec::new();
        // Counted as they are marked: no pass over the nodes for the sum.
        let mut live_nodes = 0usize;
        let mut i: u32 = 0;
        for (mark, n) in self.flags.iter_mut().zip(&self.counts) {
            if *mark & PINNED != 0 || has_outside_refs(*mark, n.others, n.internal) {
                *mark |= LIVE;
                live_nodes += 1;
                work.push(i);
            }
            i = i.wrapping_add(1);
        }
        loop {
            while let Some(i) = work.pop() {
                let node = &self.nodes[i as usize];
                let (start, len) = (node.start as usize, node.len as usize);
                for &t in self.edges.get(start..start + len).unwrap_or_default() {
                    let Some(mark) = self.flags.get_mut(t as usize) else {
                        continue;
                    };
                    if *mark & LIVE == 0 {
                        *mark |= LIVE;
                        live_nodes += 1;
                        work.push(t);
                    }
                }
            }
            // A `WeakMap` value is live when both its map and its key are.
            for c in &self.cond {
                let key_live = c.key == NONE || self.live(c.key as usize);
                if key_live && self.live(c.map as usize) && !self.live(c.value as usize) {
                    if let Some(mark) = self.flags.get_mut(c.value as usize) {
                        *mark |= LIVE;
                    }
                    live_nodes += 1;
                    work.push(c.value);
                }
            }
            if work.is_empty() {
                break;
            }
        }
        self.nodes.len() - live_nodes
    }

    /// Releases the edges and indexes once the analysis and the nested-data
    /// removal are done ([`Graph::children`] is empty afterwards: a node's
    /// range is then outside the edge list). The marks stay.
    pub fn drop_edges(&mut self) {
        self.edges = Vec::new();
        self.cond = Vec::new();
        self.obj_index = Vec::new();
        self.index = FxHashMap::default();
        self.counts = Vec::new();
    }

    /// Children of `i` (edges plus the `WeakMap` values it holds or keys).
    pub fn children(&self, i: u32) -> impl Iterator<Item = u32> + '_ {
        let n = &self.nodes[i as usize];
        let (s, l) = (n.start as usize, n.len as usize);
        let own = self.edges.get(s..s + l).unwrap_or_default();
        own.iter().copied().chain(
            self.cond
                .iter()
                .filter(move |c| c.map == i || c.key == i)
                .map(|c| c.value),
        )
    }
}

/// [`Graph::obj_node`] over the index alone (while `cond` is borrowed).
fn obj_node_in(obj_index: &[u32], h: u32) -> Option<u32> {
    obj_index.get(h as usize).copied().filter(|&i| i != NONE)
}

/// Whether a node has references from outside the candidate subgraph
/// (it is then live): `others` handles besides the graph's, `internal` of
/// them from candidate nodes.
#[inline]
fn has_outside_refs(flags: u8, others: u32, internal: u32) -> bool {
    // An object reached through no edge (found in the object store):
    // every reference to it is external. Fewer handles than edges would
    // mean an edge was enumerated twice: live.
    flags & REACHED == 0 || others != internal
}
