// buzzs-rust-php-project - Copyright (c) 2025-2026 David Buzz <davidbuzz@gmail.com>
// Licensed under the BSD-3-Clause, the license PHP master uses; see LICENSE.
//! Reading and writing places (`$a[..]->b[..]`): the equivalent of Zend's
//! `FETCH_DIM_*`, `FETCH_OBJ_*`, `ASSIGN_DIM`, `ASSIGN_OBJ`, `UNSET_*`,
//! `ISSET_ISEMPTY_*` handlers.
//!
//! A write walks the place in one go: inside one container (a variable, a
//! property slot, a static property or a reference) the walk works on
//! `&mut Value` and collects diagnostics; when it reaches an object it
//! releases all borrows and continues with the object handlers, which may
//! run user code (`__get`, `offsetGet`, ...).
//!
//! php-src runs one opcode per segment (`FETCH_DIM_W`, ..., `ASSIGN_DIM`)
//! and raises each diagnostic where it arises, holding the container
//! (`GC_ADDREF`) and giving the write up when the error handler freed or
//! shared it. Collected diagnostics are only equivalent while no error
//! handler can observe them: with one set, a walk stops at each such
//! diagnostic (`Step::Pause`), the VM raises it with the container held and
//! walks again past it ([`Vm::run_live`]). Undefined compiled variables
//! used as offsets or as the assigned value are read when the walk gets to
//! them, as the opcodes read their operands ([`PendingRead`]).

use crate::exec::KeyCtx;
use crate::object;
use crate::opcode::{FetchMode, IncDecKind, Operand, PlaceBase, PlaceId, Seg};
use crate::vm::{Throw, VarStore, Vm, VmResult};
use php_runtime::{ops, ArrayKey, BinaryOp, Obj, PhpRef, Value, ZArray, ZStr};
use std::cell::RefCell;
use std::rc::Rc;

/// A resolved place base.
#[derive(Clone)]
pub(crate) enum RBase {
    /// A variable slot.
    Var(Rc<RefCell<VarStore>>, usize),
    /// `$this`.
    This(Option<Obj>),
    /// A static property slot.
    Cell(Rc<RefCell<Value>>),
    /// A temporary value.
    Value(Value),
    /// `$GLOBALS`.
    Globals,
    /// An undefined variable-variable (read contexts).
    Undef(ZStr),
}

/// Entries of [`Vm::global_slot_cache`] (a power of two).
pub(crate) const GLOBAL_SLOT_CACHE: usize = 64;

/// A generic write to a place that starts at compiled variable `cv` of
/// the running function `frame` (see [`Vm::with_cv_symtab`]).
pub(crate) struct CvRoot {
    frame: usize,
    cv: u32,
    /// The place goes into the variable (`$a[...]`, `$a->...`): a plain
    /// array or object may be wrapped for the walk. A write to the
    /// variable itself walks it only when it already is a reference.
    wrap: bool,
    /// Set once the walk reaches the variable ([`Vm::resolve_base`]).
    walk: Option<CvWalk>,
}

/// The walk of a [`CvRoot`] on `store`, a private table holding only the
/// variable, instead of on the function's symbol table. While it runs, the
/// variable is the reference `r`, so whatever looks at the function's
/// variables meanwhile (a backtrace's arguments) sees the value being
/// written.
pub(crate) struct CvWalk {
    /// The function's own table, which identifies the frame afterwards.
    vars: *const RefCell<VarStore>,
    store: Rc<RefCell<VarStore>>,
    r: PhpRef,
    /// `r` was made for the walk: the variable held a plain array or
    /// object, which no symbol table and no other variable can reach.
    wrapped: bool,
}

/// A resolved path segment.
#[derive(Clone, Debug)]
pub(crate) enum RSeg {
    /// `[key]`
    Dim(Value),
    /// `[]`
    Append,
    /// `->name`
    Prop(ZStr),
}

/// A write operation applied at the end of a path.
pub(crate) enum WOp {
    Assign(Value),
    /// Read the current value for a read-modify-write (creates missing
    /// elements with a warning).
    FetchRw,
    /// `place op= v` (`ZEND_ASSIGN_DIM_OP` / `ZEND_ASSIGN_OBJ_OP`): walks
    /// like [`WOp::FetchRw`] once, so offset diagnostics and object
    /// handlers run a single time. Object handlers at the last segment
    /// read, operate and write back themselves; a final slot inside a
    /// container yields its current value (`Step::Done`) and is then
    /// written by a silent second walk (see `Vm::finish_native`).
    AssignOp(BinaryOp, Value),
    MakeRef,
    AssignRef(Value),
    Unset,
    /// `place = $cv` where `$cv` is a reference to null, false or an array:
    /// `ZEND_ASSIGN_DIM` reads `OP_DATA` only after fetching the container
    /// for writing, so the value is read at the final slot (see
    /// [`Vm::place_assign_cv_ref`]). Boxed: rare, and `WOp` stays small.
    AssignDeref(Box<DeferredRead>),
}

/// The source of a [`WOp::AssignDeref`] and what the walk found.
pub(crate) struct DeferredRead {
    /// The reference whose value is assigned.
    src: PhpRef,
    hit: DeferredHit,
}

/// Where a [`WOp::AssignDeref`] walk left the assignment.
enum DeferredHit {
    /// Not reached the final slot, or the value was assigned there (the
    /// source was not a container on the walked path).
    None,
    /// The source is a container on the path (its borrow is held by the
    /// walk): a placeholder was stored in the final slot, which then
    /// receives a handle to the source's array (`$t = &$a; $a[] = $t;`).
    Slot,
    /// As `Slot`, but the final slot is this reference: the source's value
    /// is assigned through it once the walk has released its borrows.
    Ref(PhpRef),
}

impl WOp {
    /// Turns a [`WOp::AssignDeref`] into a plain assignment of the source's
    /// current value (an object handler continues the walk, or the walk is
    /// over), releasing the source handle without making it a GC root.
    pub(crate) fn resolve_deferred(self) -> WOp {
        match self {
            WOp::AssignDeref(d) => {
                let DeferredRead { src, hit } = *d;
                if let DeferredHit::Ref(r) = hit {
                    crate::gc::release_duplicate(Value::Reference(r));
                }
                let v = src.try_borrow().map_or(Value::Null, |b| b.clone());
                crate::gc::release_duplicate(Value::Reference(src));
                WOp::Assign(v)
            }
            other => other,
        }
    }
}

/// How a static property base is fetched for writing, for the check of
/// its set visibility (`zend_asymmetric_property_has_set_access`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StaticWrite {
    /// `ZEND_ASSIGN_STATIC_PROP`: "Cannot modify".
    Assign,
    /// Compound assignment, increment/decrement, reference binding or
    /// taking: "Cannot indirectly modify".
    Indirect,
    /// `FETCH_STATIC_PROP_W`/`_RW` of a container: an object passes.
    Fetch,
    /// `FETCH_STATIC_PROP_UNSET` of a container: an object or an
    /// uninitialized property passes.
    FetchUnset,
}

/// The operation a place is resolved for (see [`StaticWrite`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaceOp {
    Other,
    Assign,
    Unset,
}

/// Deferred diagnostics: (level bits, message).
type Diags = Vec<(i64, String)>;

/// Errors raised inside a container walk (converted to exceptions later).
pub(crate) enum WalkErr {
    Error(String),
    TypeError(String),
    /// A string offset is assigned an object: its string conversion
    /// (`zval_try_get_string_func`, maybe `__toString()`) needs the VM, so
    /// the walk stops after the offset checks and the caller converts the
    /// value and writes its first byte with a silent second walk.
    StringOffsetObject,
}

/// Result of walking inside a container.
pub(crate) enum Step {
    /// Finished with a result value.
    Done(Value),
    /// Reached an object at segment `i` (or the value of a property).
    Obj(Obj, usize),
    /// A container error.
    Err(WalkErr),
    /// An assignment through a typed reference, performed once the walk
    /// has released its borrows (`zend_assign_to_typed_ref`).
    TypedRefAssign(PhpRef, Value),
    /// A walk in [`Mode::Live`] stopped at a diagnostic for the VM to
    /// raise ([`Vm::run_live`]). Boxed: rare, and `Step` stays small.
    Pause(Box<Pause>),
}

/// Where a live walk stopped.
pub(crate) enum Pause {
    /// A diagnostic in the middle of an array fetch, raised with the
    /// (separated) array `held` the way php-src holds it (`GC_ADDREF`).
    /// The walk resumes past phase `phase` of segment `seg`.
    Fetch {
        seg: usize,
        phase: u8,
        level: i64,
        msg: String,
        held: ZArray,
        check: Check,
    },
    /// A final assignment to a string offset that raises a diagnostic
    /// (`zend_assign_to_string_offset`).
    Str(StrAssign),
}

/// What php-src checks after a [`Pause::Fetch`] diagnostic.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Check {
    /// `zend_false_to_array_deprecated`: the fetch is given up when the
    /// handler freed the new array (`GC_DELREF(ht) == 0`); an exception
    /// does not stop the fetch, only the opcodes after it.
    False,
    /// `slow_index_convert_w`, `zend_undefined_index_write`: given up when
    /// the handler freed the array (`GC_DELREF(ht) == 0`) or captured it
    /// (any new reference: `GC_DELREF(ht) != 1`); modifying the index
    /// string in an error handler does not abandon the write.
    Index,
    /// The assigned value's "Undefined variable" warning at the final
    /// slot: given up when the array was freed (php-src would write into
    /// freed memory); the null is still stored after an exception.
    Value,
}

/// The parts of a string offset assignment raising diagnostics: the
/// separated string `held`, the offset (`undef_dim` when it is an
/// undefined variable) and the value (`undef_value` likewise).
pub(crate) struct StrAssign {
    held: ZStr,
    dim: Value,
    undef_dim: Option<ZStr>,
    value: Value,
    undef_value: Option<ZStr>,
}

/// How a walk treats the diagnostics php-src raises in the middle of a
/// fetch (see [`site`]).
#[derive(Clone, Copy)]
enum Mode {
    /// Collect them for the caller to raise afterwards: nothing can observe
    /// the walk (no error handler is set).
    Defer,
    /// Stop at the first one past phase `phase` of segment `seg` (the
    /// earlier ones were raised already).
    Live { seg: usize, phase: u8 },
    /// The error handler threw at phase `phase` of segment `seg`: that
    /// fetch completes as php-src's does with `EG(exception)` set (later
    /// diagnostics are not raised, an offset conversion gives the fetch up).
    Unwind { seg: usize, phase: u8 },
}

/// What a walk does at a diagnostic (see [`Mode`]).
#[derive(Clone, Copy)]
enum Site {
    Defer,
    Skip,
    Pause,
    Suppressed,
}

/// The phases of one segment's fetch, in php-src's order: the `false`
/// container's conversion, the undefined-variable offset, its conversion,
/// the missing key of a read-modify-write, the assigned value, a string
/// offset assignment.
const PH_FALSE: u8 = 1;
const PH_UNDEF: u8 = 2;
const PH_KEY: u8 = 3;
const PH_UNDEF_KEY: u8 = 4;
const PH_VALUE: u8 = 5;
const PH_STR: u8 = 6;

thread_local! {
    /// The mode of the walk in progress; set by [`Vm::run_live`] around each
    /// walk it runs, `Defer` otherwise (user code never runs inside a walk).
    static MODE: std::cell::Cell<Mode> = const { std::cell::Cell::new(Mode::Defer) };
}

/// What the walk does at the diagnostic of `phase` of segment `i`.
#[cold]
fn site(i: usize, phase: u8) -> Site {
    match MODE.with(std::cell::Cell::get) {
        Mode::Defer => Site::Defer,
        Mode::Live { seg, phase: p } | Mode::Unwind { seg, phase: p } if (i, phase) <= (seg, p) => {
            Site::Skip
        }
        Mode::Live { .. } => Site::Pause,
        Mode::Unwind { .. } => Site::Suppressed,
    }
}

/// Runs `f` (a walk) in `mode`.
fn with_mode<R>(mode: Mode, f: impl FnOnce() -> R) -> R {
    let saved = MODE.with(|m| m.replace(mode));
    let r = f();
    MODE.with(|m| m.set(saved));
    r
}

/// A [`Pause::Fetch`] step.
fn pause(seg: usize, phase: u8, level: i64, msg: String, held: ZArray, check: Check) -> Step {
    Step::Pause(Box::new(Pause::Fetch {
        seg,
        phase,
        level,
        msg,
        held,
        check,
    }))
}

/// The segment index of a [`PendingRead`] of the assigned value.
const VALUE: usize = usize::MAX;

/// An undefined compiled variable a write walk reads when it gets to it:
/// the offset of segment `seg` (`$a[$u][...] = v`: `FETCH_DIM_W` and
/// `ASSIGN_DIM` read it once they have the container, so its warning
/// follows the earlier fetches' diagnostics and is not raised for a
/// scalar container), or ([`VALUE`]) the assigned value (`ASSIGN_DIM` and
/// `ASSIGN_OBJ` read `OP_DATA` once they have the slot or the object).
/// Registered for the walk over the `len` segments at `segs` (see [`Segs`]).
struct PendingRead {
    segs: *const RSeg,
    len: usize,
    seg: usize,
    name: ZStr,
}

thread_local! {
    /// The pending reads of the write walks in progress.
    static PENDING_READS: RefCell<Vec<PendingRead>> = const { RefCell::new(Vec::new()) };
}

/// Whether any write walk has a pending read.
#[inline]
fn reads_pending() -> bool {
    PENDING_READS.with(|p| p.try_borrow().is_ok_and(|p| !p.is_empty()))
}

/// Takes the name of the variable pending for segment `seg` ([`VALUE`]:
/// the assigned value) of the walk over `segs`.
#[cold]
fn take_pending(segs: &[RSeg], seg: usize) -> Option<ZStr> {
    PENDING_READS.with(|p| {
        let mut p = p.try_borrow_mut().ok()?;
        let at = p.iter().position(|r| {
            std::ptr::eq(r.segs, segs.as_ptr()) && r.len == segs.len() && r.seg == seg
        })?;
        Some(p.swap_remove(at).name)
    })
}

/// A place's resolved segments ([`Vm::resolve_place`]); the reads pending
/// for a walk over them are dropped with them.
pub(crate) struct Segs {
    v: Vec<RSeg>,
    reads: bool,
}

impl Segs {
    /// Registers a pending read of `name` for segment `seg` ([`VALUE`]: the
    /// assigned value) of a walk over these segments.
    fn add_read(&mut self, seg: usize, name: ZStr) {
        let r = PendingRead {
            segs: self.v.as_ptr(),
            len: self.v.len(),
            seg,
            name,
        };
        if PENDING_READS
            .with(|p| p.try_borrow_mut().map(|mut p| p.push(r)))
            .is_ok()
        {
            self.reads = true;
        }
    }
}

impl std::ops::Deref for Segs {
    type Target = [RSeg];

    fn deref(&self) -> &[RSeg] {
        &self.v
    }
}

impl Drop for Segs {
    fn drop(&mut self) {
        if self.reads {
            let ptr = self.v.as_ptr();
            let _ = PENDING_READS.try_with(|p| {
                if let Ok(mut p) = p.try_borrow_mut() {
                    p.retain(|r| !std::ptr::eq(r.segs, ptr));
                }
            });
        }
    }
}

/// What [`Vm::run_live`] leaves to its caller.
enum Live {
    /// The walk got to its end: the step to finish and the diagnostics it
    /// collected.
    Walked(Step, Diags),
    /// The operation is over (given up, continued on a temporary, or
    /// thrown).
    Over(VmResult<Value>),
}

const E_WARNING: i64 = 2;
const E_DEPRECATED: i64 = 8192;

const FALSE_TO_ARRAY: &str = "Automatic conversion of false to array is deprecated";

/// Converts an array key for a write/read context without the VM (pure);
/// diagnostics are deferred.
pub(crate) fn key_of(v: &Value, diags: &mut Diags, ctx: KeyCtx) -> Result<ArrayKey, WalkErr> {
    Ok(match v {
        Value::Int(i) => ArrayKey::Int(*i),
        Value::Str(s) => ArrayKey::from_zstr(s),
        Value::Null => {
            diags.push((
                E_DEPRECATED,
                "Using null as an array offset is deprecated, use an empty string instead".into(),
            ));
            ArrayKey::Str(ZStr::empty())
        }
        Value::Bool(b) => ArrayKey::Int(i64::from(*b)),
        Value::Float(d) => {
            let d = *d;
            if d.is_nan() || d.is_infinite() {
                diags.push((
                    E_WARNING,
                    format!(
                        "The float {} is not representable as an int, cast occurred",
                        php_runtime::float::format_double(d, -1, false)
                    ),
                ));
                ArrayKey::Int(0)
            } else {
                let l = php_runtime::convert::dval_to_lval_silent(d);
                #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
                if !crate::typecheck::fits_long(d) {
                    diags.push((
                        E_WARNING,
                        format!(
                            "The float {} is not representable as an int, cast occurred",
                            php_runtime::float::format_double(d, -1, false)
                        ),
                    ));
                } else if l as f64 != d {
                    diags.push((
                        E_DEPRECATED,
                        format!(
                            "Implicit conversion from float {} to int loses precision",
                            php_runtime::float::format_double(d, -1, false)
                        ),
                    ));
                }
                ArrayKey::Int(l)
            }
        }
        Value::Resource(r) => {
            diags.push((
                E_WARNING,
                format!(
                    "Resource ID#{} used as offset, casting to integer ({})",
                    r.handle(),
                    r.handle()
                ),
            ));
            ArrayKey::Int(r.handle())
        }
        Value::Reference(r) => return key_of(&r.get(), diags, ctx),
        Value::Array(_) | Value::Object(_) => {
            let tn = crate::typecheck::given_type(v);
            return Err(WalkErr::TypeError(match ctx {
                KeyCtx::Isset => format!("Cannot access offset of type {tn} in isset or empty"),
                KeyCtx::Unset => format!("Cannot unset offset of type {tn} on array"),
                // Array literals too (`zend_illegal_array_offset_access`);
                // "Illegal offset type" is only the compile-time error.
                _ => format!("Cannot access offset of type {tn} on array"),
            }));
        }
    })
}

fn undefined_key_msg(k: &ArrayKey) -> String {
    match k {
        ArrayKey::Int(i) => format!("Undefined array key {i}"),
        ArrayKey::Str(s) => format!("Undefined array key \"{}\"", s.to_string_lossy()),
    }
}

/// The handles of a refcounted string or array (0 for other values).
fn shared_handles(v: &Value) -> usize {
    match v {
        Value::Str(s) => s.refcount(),
        Value::Array(a) => a.refcount(),
        _ => 0,
    }
}

/// String offset conversion (`zend_check_string_offset`). `unset` is the
/// `BP_VAR_UNSET` fetch type: no leading-numeric warning, and illegal
/// offsets report "Cannot unset string offsets".
fn string_offset(dim: &Value, diags: &mut Diags, unset: bool) -> Result<i64, WalkErr> {
    match dim {
        Value::Int(i) => Ok(*i),
        Value::Str(s) => {
            if let Some(n) = php_runtime::numeric::is_numeric_string_ex(s.as_bytes(), true) {
                if let php_runtime::numeric::Number::Int(i) = n.value {
                    if n.trailing_data && !unset {
                        diags.push((
                            E_WARNING,
                            format!("Illegal string offset \"{}\"", s.to_string_lossy()),
                        ));
                    }
                    return Ok(i);
                }
            }
            if unset {
                return Err(WalkErr::Error("Cannot unset string offsets".into()));
            }
            Err(WalkErr::TypeError(
                "Cannot access offset of type string on string".to_string(),
            ))
        }
        Value::Float(d) => {
            diags.push((E_WARNING, "String offset cast occurred".into()));
            Ok(php_runtime::convert::dval_to_lval_silent(*d))
        }
        Value::Null | Value::Bool(_) => {
            diags.push((E_WARNING, "String offset cast occurred".into()));
            Ok(match dim {
                Value::Bool(true) => 1,
                _ => 0,
            })
        }
        Value::Reference(r) => string_offset(&r.get(), diags, unset),
        _ if unset => Err(WalkErr::Error("Cannot unset string offsets".into())),
        other => Err(WalkErr::TypeError(format!(
            "Cannot access offset of type {} on string",
            crate::typecheck::given_type(other)
        ))),
    }
}

/// String offset conversion of a quiet read (`zend_fetch_dimension_address_read`
/// with `BP_VAR_IS`): casts are silent and a non-numeric string gives
/// `None`, but a leading-numeric string still warns and an illegal offset
/// type still throws.
fn string_offset_is(dim: &Value, diags: &mut Diags) -> Result<Option<i64>, WalkErr> {
    use php_runtime::numeric::Number;
    match dim {
        Value::Int(i) => Ok(Some(*i)),
        Value::Str(s) => match php_runtime::numeric::is_numeric_string_ex(s.as_bytes(), true) {
            Some(n) => match n.value {
                Number::Int(i) => {
                    if n.trailing_data {
                        diags.push((
                            E_WARNING,
                            format!("Illegal string offset \"{}\"", s.to_string_lossy()),
                        ));
                    }
                    Ok(Some(i))
                }
                Number::Float(_) => Ok(None),
            },
            None => Ok(None),
        },
        Value::Float(d) => Ok(Some(php_runtime::convert::dval_to_lval_silent(*d))),
        Value::Bool(b) => Ok(Some(i64::from(*b))),
        Value::Null => Ok(Some(0)),
        Value::Reference(r) => string_offset_is(&r.get(), diags),
        other => Err(WalkErr::TypeError(format!(
            "Cannot access offset of type {} on string",
            crate::typecheck::given_type(other)
        ))),
    }
}

/// A string container in a write walk. Only a final plain assignment
/// writes (`zend_assign_to_string_offset`); every other context checks the
/// offset first (`zend_check_string_offset`, with its warnings) and then
/// throws the context's error: `zend_use_new_element_for_string`,
/// `ZEND_UNSET_DIM`'s "Cannot unset string offsets" and
/// `zend_wrong_string_offset_error`.
#[cold]
fn string_container_step(
    v: &mut Value,
    segs: &[RSeg],
    i: usize,
    wop: &mut WOp,
    diags: &mut Diags,
) -> Step {
    let last = i + 1 == segs.len();
    let unset = matches!(wop, WOp::Unset);
    if unset && last {
        return Step::Err(WalkErr::Error("Cannot unset string offsets".into()));
    }
    let RSeg::Dim(dim) = &segs[i] else {
        return Step::Err(WalkErr::Error(
            "[] operator not supported for strings".into(),
        ));
    };
    // `zend_check_string_offset` reads an undefined-variable offset (its
    // warning, then "String offset cast occurred").
    let undef_dim = if dim.is_null() {
        take_pending(segs, i)
    } else {
        None
    };
    if last {
        let val = match wop {
            WOp::Assign(val) => Some(val.clone()),
            // The source is null, false or an array (one holding this
            // string when the walk holds its borrow): converting any array
            // to a string offset value behaves alike.
            WOp::AssignDeref(d) => Some(
                d.src
                    .try_borrow()
                    .map_or_else(|| Value::Array(ZArray::new()), |b| b.clone()),
            ),
            _ => None,
        };
        if let Some(val) = val {
            let undef_value = if matches!(wop, WOp::Assign(Value::Null)) {
                take_pending(segs, VALUE)
            } else {
                None
            };
            let quiet = undef_dim.is_none()
                && undef_value.is_none()
                && matches!(dim, Value::Int(_))
                && matches!(&val, Value::Str(s) if s.len() == 1);
            if !quiet && matches!(site(i, PH_STR), Site::Pause) {
                // `zend_assign_to_string_offset` separates the string and
                // holds it across each diagnostic: the VM raises them
                // (`Vm::live_string_assign`).
                if let Value::Str(s) = v {
                    if s.is_interned() || s.refcount() != 1 {
                        *s = ZStr::new(s.as_bytes().to_vec());
                    }
                    return Step::Pause(Box::new(Pause::Str(StrAssign {
                        held: s.clone(),
                        dim: dim.clone(),
                        undef_dim,
                        value: val,
                        undef_value,
                    })));
                }
            }
            if let Some(name) = undef_dim {
                diags.push((E_WARNING, crate::exec::undefined_var_msg(&name)));
            }
            return match assign_string_offset(v, dim, &val, diags, undef_value.as_ref()) {
                Ok(r) => Step::Done(r),
                Err(e) => Step::Err(e),
            };
        }
    }
    if let Some(name) = undef_dim {
        diags.push((E_WARNING, crate::exec::undefined_var_msg(&name)));
    }
    if let Err(e) = string_offset(dim, diags, unset) {
        return Step::Err(e);
    }
    let msg = if last {
        match wop {
            WOp::FetchRw => "Cannot increment/decrement string offsets",
            WOp::MakeRef | WOp::AssignRef(_) => "Cannot create references to/from string offsets",
            _ => "Cannot use assign-op operators with string offsets",
        }
    } else if matches!(segs[i + 1], RSeg::Prop(_)) {
        // FETCH_DIM_W/RW/UNSET feeding a property or dimension fetch.
        "Cannot use string offset as an object"
    } else {
        "Cannot use string offset as an array"
    };
    Step::Err(WalkErr::Error(msg.into()))
}

/// The string form of a value written to a string offset; its first byte
/// is stored (`zend_assign_to_string_offset`).
fn string_offset_value_bytes(val: &Value) -> Vec<u8> {
    match val {
        Value::Str(v) => v.as_bytes().to_vec(),
        Value::Int(i) => i.to_string().into_bytes(),
        Value::Bool(true) => b"1".to_vec(),
        Value::Bool(false) | Value::Null => Vec::new(),
        Value::Float(f) => php_runtime::float::format_double(*f, 17, false).into_bytes(),
        Value::Array(_) => b"Array".to_vec(),
        Value::Resource(r) => format!("Resource id #{}", r.handle()).into_bytes(),
        // Objects are converted by `__toString()` on the retry path
        // (`assign_string_offset` reports `StringOffsetObject`).
        Value::Object(_) | Value::Reference(_) => b"A".to_vec(),
    }
}

/// Writes one byte of a string (`zend_assign_to_string_offset`);
/// `undef_value` names the undefined variable the value was read from (its
/// warning comes after the offset checks).
fn assign_string_offset(
    s: &mut Value,
    dim: &Value,
    val: &Value,
    diags: &mut Diags,
    undef_value: Option<&ZStr>,
) -> Result<Value, WalkErr> {
    let Value::Str(str) = s else {
        return Ok(Value::Null);
    };
    let offset = string_offset(dim, diags, false)?;
    let len = i64::try_from(str.len()).unwrap_or(i64::MAX);
    if offset < -len {
        diags.push((E_WARNING, format!("Illegal string offset {offset}")));
        return Ok(Value::Null);
    }
    let offset = if offset < 0 { offset + len } else { offset };
    if let Some(name) = undef_value {
        diags.push((E_WARNING, crate::exec::undefined_var_msg(name)));
    }
    let bytes: Vec<u8> = match val.deref() {
        Value::Str(v) => v.as_bytes().to_vec(),
        Value::Int(i) => i.to_string().into_bytes(),
        Value::Bool(true) => b"1".to_vec(),
        Value::Bool(false) | Value::Null => Vec::new(),
        Value::Float(f) => php_runtime::float::format_double(f, 17, false).into_bytes(),
        // `zval_try_get_string_func`.
        Value::Array(_) => {
            diags.push((E_WARNING, "Array to string conversion".into()));
            b"Array".to_vec()
        }
        Value::Resource(r) => format!("Resource id #{}", r.handle()).into_bytes(),
        Value::Object(_) | Value::Reference(_) => return Err(WalkErr::StringOffsetObject),
    };
    if bytes.is_empty() {
        return Err(WalkErr::Error(
            "Cannot assign an empty string to a string offset".into(),
        ));
    }
    if bytes.len() != 1 {
        diags.push((
            E_WARNING,
            "Only the first byte will be assigned to the string offset".into(),
        ));
    }
    let c = bytes[0];
    // In place when the variable holds the only handle
    // (`zend_assign_to_string_offset` separates a shared string only).
    str.set_byte(usize::try_from(offset).unwrap_or(0), c);
    Ok(Value::Str(ZStr::from_byte(c)))
}

/// The walk inside one container.
#[allow(clippy::too_many_lines)]
pub(crate) fn descend(
    v: &mut Value,
    segs: &[RSeg],
    i: usize,
    wop: &mut WOp,
    diags: &mut Diags,
    undef: bool,
) -> Step {
    if i == segs.len() {
        return apply_final(v, wop);
    }
    if let Value::Reference(r) = v {
        // Walked through the container's own handle: a temporary clone
        // released afterwards would be buffered as a GC possible root
        // (php-src never touches the count here).
        let r: &PhpRef = r;
        if r.try_borrow().is_none() {
            return Step::Err(WalkErr::Error(
                "Cannot modify a value while it is being iterated".into(),
            ));
        }
        let dim = !matches!(segs[i], RSeg::Prop(_));
        let last = i + 1 == segs.len();
        // `ZEND_ASSIGN_DIM_OP` works on the referenced value unchecked; the
        // fetches and `ZEND_ASSIGN_DIM` verify a typed reference first.
        if dim && !matches!(wop, WOp::Unset) && !(last && matches!(wop, WOp::AssignOp(..))) {
            if let Some(msg) = crate::typed_refs::walk_auto_init_error(r) {
                return Step::Err(WalkErr::TypeError(msg));
            }
        }
        let mut b = r.borrow_mut();
        // A fetch (`zend_fetch_dimension_address`: FETCH_DIM_W, _RW and
        // _UNSET) makes an array of a referenced `false` without the
        // false-to-array deprecation, which only `ZEND_ASSIGN_DIM`,
        // `ZEND_ASSIGN_DIM_OP` and `ZEND_UNSET_DIM` raise.
        let fetch = !last || matches!(wop, WOp::FetchRw | WOp::MakeRef | WOp::AssignRef(_));
        if dim && fetch && matches!(*b, Value::Bool(false)) {
            if matches!(wop, WOp::Unset) {
                return Step::Done(Value::Null);
            }
            *b = Value::Array(ZArray::new());
        }
        return descend(&mut b, segs, i, wop, diags, false);
    }
    let last = i + 1 == segs.len();
    let is_unset = matches!(wop, WOp::Unset);
    match &segs[i] {
        RSeg::Prop(name) => match v {
            Value::Object(o) => Step::Obj(o.clone(), i),
            other => {
                if is_unset {
                    return Step::Done(Value::Null);
                }
                let tn = other.value_name();
                let tn = if undef { "null".to_string() } else { tn };
                // `zend_throw_non_object_error`: ASSIGN_OBJ and
                // ASSIGN_OBJ_OP "assign", PRE/POST_INC/DEC_OBJ
                // "increment/decrement", FETCH_OBJ_W/RW "modify".
                if last
                    && matches!(
                        wop,
                        WOp::Assign(_) | WOp::AssignOp(..) | WOp::AssignDeref(_)
                    )
                {
                    // ZEND_ASSIGN_OBJ reads OP_DATA before it looks at the
                    // object.
                    if matches!(wop, WOp::Assign(Value::Null)) {
                        push_pending_value(segs, diags);
                    }
                    Step::Err(WalkErr::Error(format!(
                        "Attempt to assign property \"{}\" on {tn}",
                        name.to_string_lossy()
                    )))
                } else if last && matches!(wop, WOp::FetchRw) {
                    // A final FetchRw walk is the read of `++`/`--`
                    // (compound assignments walk with `WOp::AssignOp`).
                    Step::Err(WalkErr::Error(format!(
                        "Attempt to increment/decrement property \"{}\" on {tn}",
                        name.to_string_lossy()
                    )))
                } else {
                    Step::Err(WalkErr::Error(format!(
                        "Attempt to modify property \"{}\" on {tn}",
                        name.to_string_lossy()
                    )))
                }
            }
        },
        seg @ (RSeg::Dim(_) | RSeg::Append) => {
            match v {
                Value::Array(_) => {}
                Value::Null => {
                    if is_unset {
                        return Step::Done(Value::Null);
                    }
                    *v = Value::Array(ZArray::new());
                }
                Value::Bool(false) => {
                    if is_unset {
                        // UNSET_DIM / FETCH_DIM_UNSET leave `false` alone.
                        diags.push((E_DEPRECATED, FALSE_TO_ARRAY.into()));
                        return Step::Done(Value::Null);
                    }
                    *v = Value::Array(ZArray::new());
                    if let Some(step) = false_to_array_site(v, i, diags) {
                        return step;
                    }
                }
                Value::Object(o) => return Step::Obj(o.clone(), i),
                Value::Str(_) => return string_container_step(v, segs, i, wop, diags),
                _ => {
                    if is_unset {
                        return Step::Err(WalkErr::Error(
                            "Cannot unset offset in a non-array variable".into(),
                        ));
                    }
                    // `ZEND_ASSIGN_DIM_OP` reads the offset before
                    // `zend_binary_assign_op_dim_slow` throws; the fetches
                    // and `ZEND_ASSIGN_DIM` throw first.
                    if last
                        && matches!(wop, WOp::AssignOp(..))
                        && matches!(seg, RSeg::Dim(Value::Null))
                    {
                        if let Some(name) = take_pending(segs, i) {
                            diags.push((E_WARNING, crate::exec::undefined_var_msg(&name)));
                        }
                    }
                    return Step::Err(WalkErr::Error(
                        "Cannot use a scalar value as an array".into(),
                    ));
                }
            }
            let Value::Array(arr) = v else {
                return Step::Done(Value::Null);
            };
            match seg {
                RSeg::Append => {
                    let t = arr.make_mut();
                    if last {
                        match wop {
                            WOp::Assign(val) => {
                                // ZEND_ASSIGN_DIM reads OP_DATA before the
                                // insertion.
                                if val.is_null() {
                                    push_pending_value(segs, diags);
                                }
                                let val = std::mem::take(val);
                                if t.push(val.clone()).is_err() {
                                    return Step::Err(WalkErr::Error(
                                        "Cannot add element to the array as the next element is already occupied".into(),
                                    ));
                                }
                                Step::Done(val)
                            }
                            WOp::AssignRef(r) => {
                                let r = std::mem::take(r);
                                if t.push(r.clone()).is_err() {
                                    return Step::Err(WalkErr::Error(
                                        "Cannot add element to the array as the next element is already occupied".into(),
                                    ));
                                }
                                Step::Done(r)
                            }
                            _ => {
                                // ZEND_ASSIGN_DIM_OP reads OP_DATA once it
                                // has the new element.
                                if matches!(wop, WOp::AssignOp(..)) && reads_pending() {
                                    push_pending_value(segs, diags);
                                }
                                // The key `push_mut` uses (`nNextFreeElement`).
                                let slot_key = t.next_index();
                                match t.push_mut(Value::Null) {
                                    Ok(slot) => {
                                        if matches!(wop, WOp::AssignOp(..)) {
                                            record_append(i, slot_key);
                                        }
                                        apply_final(slot, wop)
                                    }
                                    Err(_) => Step::Err(WalkErr::Error(
                                        "Cannot add element to the array as the next element is already occupied".into(),
                                    )),
                                }
                            }
                        }
                    } else {
                        let slot_key = t.next_index();
                        match t.push_mut(Value::Null) {
                            Ok(slot) => {
                                if matches!(wop, WOp::AssignOp(..)) {
                                    record_append(i, slot_key);
                                }
                                descend(slot, segs, i + 1, wop, diags, true)
                            }
                            Err(_) => {
                                Step::Err(WalkErr::Error(
                                        "Cannot add element to the array as the next element is already occupied".into(),
                                    ))
                            }
                        }
                    }
                }
                RSeg::Dim(k) => {
                    let key = match k {
                        Value::Int(n) => ArrayKey::Int(*n),
                        Value::Str(s) => ArrayKey::from_zstr(s),
                        _ => match slow_key(arr, k, segs, i, is_unset, diags) {
                            Ok(key) => key,
                            Err(step) => return step,
                        },
                    };
                    if is_unset {
                        if last {
                            if arr.contains_key(&key) {
                                // `zval_ptr_dtor` of the removed element:
                                // a possible root when it stays referenced.
                                let old = arr.make_mut().remove(&key);
                                if let Some(v) = &old {
                                    crate::gc::possible_root(v);
                                }
                                drop(old);
                            }
                            return Step::Done(Value::Null);
                        }
                        if !arr.contains_key(&key) {
                            return Step::Done(Value::Null);
                        }
                        let t = arr.make_mut();
                        let Some(slot) = t.get_mut(&key) else {
                            return Step::Done(Value::Null);
                        };
                        return descend(slot, segs, i + 1, wop, diags, false);
                    }
                    let t = arr.make_mut();
                    let rw = matches!(wop, WOp::FetchRw | WOp::AssignOp(..));
                    // A read-modify-write warns about a missing key before
                    // adding it (`zend_undefined_index_write`).
                    let found = if rw { t.position_of(&key) } else { None };
                    if rw && found.is_none() {
                        match site(i, PH_UNDEF_KEY) {
                            Site::Defer => diags.push((E_WARNING, undefined_key_msg(&key))),
                            Site::Skip => {}
                            // `EG(exception)`: the key is not added.
                            Site::Suppressed => return Step::Done(Value::Null),
                            Site::Pause => return undefined_key_pause(arr, &key, i),
                        }
                    }
                    let (slot, existed) = match found {
                        Some(pos) => match t.at_mut(pos) {
                            Some(slot) => (slot, true),
                            None => return Step::Done(Value::Null),
                        },
                        None => {
                            let (slot, inserted) = t.lookup_ex(key.clone());
                            (slot, !inserted)
                        }
                    };
                    if last {
                        match wop {
                            WOp::AssignRef(r) => {
                                let r = std::mem::take(r);
                                *slot = r.clone();
                                Step::Done(r)
                            }
                            _ => {
                                if matches!(wop, WOp::Assign(Value::Null)) && reads_pending() {
                                    return value_site(arr, segs, i, &key, wop, diags);
                                }
                                // ZEND_ASSIGN_DIM_OP reads OP_DATA once it
                                // has the slot.
                                if matches!(wop, WOp::AssignOp(..)) && reads_pending() {
                                    push_pending_value(segs, diags);
                                }
                                if rw && existed && diags.is_empty() {
                                    if let Some(step) = concat_in_place(slot, segs, wop) {
                                        return step;
                                    }
                                }
                                let step = apply_final(slot, wop);
                                if !existed && rw {
                                    watch_rw_container(arr);
                                }
                                step
                            }
                        }
                    } else {
                        descend(slot, segs, i + 1, wop, diags, !existed)
                    }
                }
                RSeg::Prop(_) => Step::Done(Value::Null),
            }
        }
    }
}

/// The deprecation of a `false` container just made the (new, empty)
/// array `v` (`zend_false_to_array_deprecated`, raised with the array
/// held). `Some` stops the walk with that step.
#[cold]
#[inline(never)]
fn false_to_array_site(v: &mut Value, i: usize, diags: &mut Diags) -> Option<Step> {
    match site(i, PH_FALSE) {
        Site::Defer => {
            diags.push((E_DEPRECATED, FALSE_TO_ARRAY.into()));
            None
        }
        Site::Skip | Site::Suppressed => None,
        Site::Pause => {
            let Value::Array(a) = v else {
                return None;
            };
            // Its own table (not the shared empty one), whose count then
            // tells whether the error handler freed it.
            a.make_mut();
            Some(pause(
                i,
                PH_FALSE,
                E_DEPRECATED,
                FALSE_TO_ARRAY.into(),
                a.clone(),
                Check::False,
            ))
        }
    }
}

/// The key of an offset `k` of segment `i` that is neither an int nor a
/// string, in the array `arr` (`slow_index_convert_w`): an undefined
/// variable is read first (its warning), then the offset converts (null,
/// float and resource diagnostics), each raised as the walk's mode says
/// with the separated array held. `Err` ends the walk with that step.
#[cold]
#[inline(never)]
fn slow_key(
    arr: &mut ZArray,
    k: &Value,
    segs: &[RSeg],
    i: usize,
    is_unset: bool,
    diags: &mut Diags,
) -> Result<ArrayKey, Step> {
    let last = i + 1 == segs.len();
    // ZEND_UNSET_DIM maps a null offset to "" without the null-offset
    // deprecation; intermediate unset fetches (FETCH_DIM_UNSET) still
    // raise it.
    if is_unset && last && k.is_null() {
        return Ok(ArrayKey::Str(ZStr::empty()));
    }
    if k.is_null() {
        if let Some(step) = undefined_offset_site(arr, segs, i, diags) {
            return Err(step);
        }
    }
    // Only the final ZEND_UNSET_DIM reports "Cannot unset offset";
    // FETCH_DIM_UNSET converts like a write fetch.
    let ctx = if is_unset && last {
        KeyCtx::Unset
    } else {
        KeyCtx::Write
    };
    let n = diags.len();
    let key = key_of(k, diags, ctx).map_err(Step::Err)?;
    if diags.len() > n {
        if let Some(step) = key_site(arr, i, diags, n) {
            return Err(step);
        }
    }
    Ok(key)
}

/// The "Undefined variable" warning of an offset of segment `i` read from
/// an undefined compiled variable (`slow_index_convert_w`'s `IS_UNDEF`);
/// nothing for a null offset of another origin. `Some` stops the walk.
#[cold]
fn undefined_offset_site(
    arr: &mut ZArray,
    segs: &[RSeg],
    i: usize,
    diags: &mut Diags,
) -> Option<Step> {
    let s = site(i, PH_UNDEF);
    if matches!(s, Site::Skip) {
        return None;
    }
    let name = take_pending(segs, i)?;
    let msg = crate::exec::undefined_var_msg(&name);
    match s {
        Site::Pause => {
            arr.make_mut();
            Some(pause(
                i,
                PH_UNDEF,
                E_WARNING,
                msg,
                arr.clone(),
                Check::Index,
            ))
        }
        // `EG(exception)`: the fetch gives up.
        Site::Suppressed => Some(Step::Done(Value::Null)),
        Site::Defer | Site::Skip => {
            diags.push((E_WARNING, msg));
            None
        }
    }
}

/// The diagnostic `key_of` raised converting an offset of segment `i` (the
/// entries of `diags` from `n` on). `Some` stops the walk.
#[cold]
fn key_site(arr: &mut ZArray, i: usize, diags: &mut Diags, n: usize) -> Option<Step> {
    match site(i, PH_KEY) {
        Site::Defer => None,
        Site::Skip => {
            diags.truncate(n);
            None
        }
        Site::Suppressed => {
            diags.truncate(n);
            Some(Step::Done(Value::Null))
        }
        Site::Pause => {
            let (level, msg) = diags.drain(n..).next()?;
            arr.make_mut();
            Some(pause(i, PH_KEY, level, msg, arr.clone(), Check::Index))
        }
    }
}

/// The live walk's stop at the "Undefined array key" warning of a
/// read-modify-write whose key is missing from `arr`
/// (`zend_undefined_index_write` / `zend_undefined_offset_write`: raised
/// before the key is added, with the array held).
#[cold]
#[inline(never)]
fn undefined_key_pause(arr: &mut ZArray, key: &ArrayKey, i: usize) -> Step {
    arr.make_mut();
    pause(
        i,
        PH_UNDEF_KEY,
        E_WARNING,
        undefined_key_msg(key),
        arr.clone(),
        Check::Index,
    )
}

/// The final assignment at `key` of the array `arr` (segment `i`) when the
/// value may be an undefined variable pending for the walk: `ZEND_ASSIGN_DIM`
/// reads `OP_DATA` once it has the slot, so its warning comes after the
/// offsets' diagnostics; then the store.
#[cold]
#[inline(never)]
fn value_site(
    arr: &mut ZArray,
    segs: &[RSeg],
    i: usize,
    key: &ArrayKey,
    wop: &mut WOp,
    diags: &mut Diags,
) -> Step {
    if let Some(name) = take_pending(segs, VALUE) {
        let msg = crate::exec::undefined_var_msg(&name);
        match site(i, PH_VALUE) {
            Site::Defer => diags.push((E_WARNING, msg)),
            Site::Pause => {
                arr.make_mut();
                return pause(i, PH_VALUE, E_WARNING, msg, arr.clone(), Check::Value);
            }
            Site::Skip | Site::Suppressed => {}
        }
    }
    let slot = arr.make_mut().lookup(key.clone());
    apply_final(slot, wop)
}

/// Collects the warning of the undefined variable pending as the assigned
/// value of the walk over `segs`, if any.
#[cold]
fn push_pending_value(segs: &[RSeg], diags: &mut Diags) {
    if let Some(name) = take_pending(segs, VALUE) {
        diags.push((E_WARNING, crate::exec::undefined_var_msg(&name)));
    }
}

/// php-src once an error handler threw at the diagnostic of phase `phase`
/// of segment `seg`'s fetch, which goes on with `EG(exception)` set:
/// converting a `false` container has no exception check, so that fetch
/// completes (an intermediate `FETCH_DIM_W` adds its key, `FETCH_DIM_RW`
/// gives up at a missing one; a final assignment stores its value, as it
/// also does after the value's own warning) and raises nothing more; the
/// opcodes after it do not run.
#[cold]
#[inline(never)]
fn unwind_live(
    segs: &[RSeg],
    wop: &mut WOp,
    walk: &mut dyn FnMut(&[RSeg], &mut WOp, &mut Diags) -> Step,
    seg: usize,
    phase: u8,
) {
    let mode = Mode::Unwind { seg, phase };
    let mut diags = Vec::new();
    let substitute = if seg + 1 < segs.len() {
        Some(if matches!(wop, WOp::FetchRw | WOp::AssignOp(..)) {
            WOp::FetchRw
        } else {
            WOp::Assign(Value::Null)
        })
    } else if matches!(wop, WOp::MakeRef | WOp::AssignRef(_)) {
        // The final `FETCH_DIM_W`: the reference is never made.
        Some(WOp::Assign(Value::Null))
    } else {
        None
    };
    let s = match substitute {
        // The walk ends at the interrupted fetch.
        Some(mut w) => with_mode(mode, || walk(&segs[..=seg], &mut w, &mut diags)),
        None => with_mode(mode, || walk(segs, wop, &mut diags)),
    };
    drop(s);
    flush_deferred_ref_assigns();
}

thread_local! {
    /// Assignments through references that were borrowed by the walk that
    /// made them (see [`apply_final`]).
    static DEFERRED_REF_ASSIGNS: std::cell::RefCell<Vec<(PhpRef, Value)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    /// The elements `[]` segments appended during the first walk of a
    /// compound assignment, as (segment index, key). `ZEND_ASSIGN_DIM_OP`
    /// with no offset inserts one null element and operates on it in place
    /// (`zend_hash_next_index_insert(ht, &EG(uninitialized_zval))`); the
    /// silent write-back walk targets those elements instead of appending
    /// again (bug32833: `$a[] .= 'x'`).
    static APPENDED: std::cell::RefCell<Vec<(usize, i64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// See `APPENDED`.
#[cold]
#[inline(never)]
fn record_append(seg: usize, key: i64) {
    APPENDED.with(|a| a.borrow_mut().push((seg, key)));
}

/// The segments for the write-back walk of a compound assignment whose
/// first walk over `segs` appended elements: each recorded `[]` becomes
/// the offset of the element it appended. `None` when nothing was
/// appended (the walk is repeated over `segs` as is).
fn take_appended_segs(segs: &[RSeg]) -> Option<Vec<RSeg>> {
    if !segs.iter().any(|s| matches!(s, RSeg::Append)) {
        return None;
    }
    let rec = APPENDED.with(|a| std::mem::take(&mut *a.borrow_mut()));
    if rec.is_empty() {
        return None;
    }
    let mut out = segs.to_vec();
    for (i, k) in rec {
        if let Some(seg @ RSeg::Append) = out.get_mut(i) {
            *seg = RSeg::Dim(Value::Int(k));
        }
    }
    Some(out)
}

thread_local! {
    /// The `ArrayAccess` object and offset a read-modify-write walk ended
    /// on (`FETCH_DIM_RW` on an object yields a temporary): set by
    /// `Vm::obj_walk`, taken by `Vm::fetch_rw`.
    static OVERLOADED_RW: std::cell::Cell<Option<(Obj, Value)>> =
        const { std::cell::Cell::new(None) };
    /// The array whose missing key a read-modify-write walk added, held
    /// across the "Undefined array key" warning like
    /// `zend_undefined_index_write`'s `GC_ADDREF`: an error handler that
    /// writes to or frees the array leaves this the only owner.
    static RW_UNDEF_WATCH: std::cell::Cell<Option<ZArray>> = const { std::cell::Cell::new(None) };
    /// Set when a `WOp::FetchRw` walk was abandoned that way, or given up
    /// when a live walk's error handler freed or shared the array (read by
    /// `Vm::fetch_rw`).
    static RW_SKIPPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The frame depth of a `$o->p =& f()` walk whose value is not a
    /// reference: set by `Vm::place_assign_ref_nonvar_prop`, taken by
    /// `Vm::assign_property_ref` once the walk reaches the property.
    static PENDING_NONVAR_REF: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    /// The exception an error handler threw at the "Undefined property"
    /// warning (or the dynamic-property deprecation) of `$o->undef++`:
    /// `zend_std_get_property_ptr_ptr` has added the null property by then
    /// and the increment still happens; `Vm::place_incdec` throws it after
    /// writing the result (see `Vm::rw_dynamic_threw`).
    static RW_THROW: std::cell::Cell<Option<Throw>> = const { std::cell::Cell::new(None) };
    /// The frame depth of the silent write-back walk of `++`/`--` (after
    /// `FETCH_DIM_RW` has reported the offsets): the container walks it
    /// reaches through objects drop their diagnostics too.
    static QUIET_AT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Whether the property reference assignment at frame depth `depth` is the
/// `ZEND_RETURNS_FUNCTION` case of [`Vm::place_assign_ref`] (a call result
/// that is not a reference); clears the mark.
pub(crate) fn take_nonvar_ref(depth: usize) -> bool {
    PENDING_NONVAR_REF.with(|c| {
        if c.get() == Some(depth) {
            c.set(None);
            true
        } else {
            false
        }
    })
}

/// The in-place append behind [`Vm::concat_assign_cv`]: `None` when place
/// `pid` is not a plain compiled variable holding a string, or `rhs` is
/// neither a string nor an int, or is a second handle of that very string
/// (`$s .= $s`: the generic path releases the operand first and extends
/// the one string in place with [`ZStr::append_self`], php-src's
/// `concat_function` with `result == op1 == op2`, so the memory limit sees
/// only the growth); otherwise the string was extended (a shared one is
/// copied first, as `zend_string_extend` requires a refcount of 1) and the
/// result is returned: the new string when `want_result`, else `Null`.
#[inline]
fn concat_cv_in_place(
    f: &crate::vm::Frame,
    pid: PlaceId,
    rhs: &Value,
    want_result: bool,
) -> Option<Value> {
    let p = &f.code.places[pid as usize];
    let (PlaceBase::Cv(cv), true) = (&p.base, p.segs.is_empty()) else {
        return None;
    };
    let idx = *cv as usize;
    let slot = f.cvs.get(idx)?;
    // SAFETY: no other reference into the slot is alive: `rhs` is an owned
    // operand or a literal, never a borrow of a variable slot.
    let val = unsafe { slot.get_mut_unchecked() };
    let Some(Value::Str(s)) = val else {
        return None;
    };
    match rhs {
        Value::Str(add) if ZStr::ptr_eq(s, add) => return None,
        Value::Str(add) => {
            if f.symtab_attached {
                f.unsync_cv_from_symtab(*cv);
            }
            s.append_str(add);
        }
        Value::Int(i) => {
            if f.symtab_attached {
                f.unsync_cv_from_symtab(*cv);
            }
            s.append_long(*i);
        }
        _ => return None,
    }
    if f.symtab_attached {
        f.sync_cv_to_symtab(*cv, &Value::Str(s.clone()));
    }
    Some(if want_result {
        Value::Str(s.clone())
    } else {
        Value::Null
    })
}

/// `++`/`--` of the int in `slot` when the result stays an int: the slot
/// is updated and the expression's value returned (`PRE/POST_INC/DEC` on
/// `IS_LONG` without overflow); `None`, nothing changed, on overflow (the
/// generic path promotes to float).
#[inline]
fn incdec_int_slot(slot: &mut i64, kind: IncDecKind) -> Option<Value> {
    let old = *slot;
    let new = if kind.is_inc() {
        old.checked_add(1)?
    } else {
        old.checked_sub(1)?
    };
    *slot = new;
    Some(Value::Int(if kind.is_post() { old } else { new }))
}

/// The int fast path of [`Vm::place_incdec`] after the generic place
/// resolution: a variable slot holding an int directly (not a reference).
#[inline]
fn incdec_var_int_fast(base: &RBase, kind: IncDecKind) -> Option<Value> {
    let RBase::Var(store, idx) = base else {
        return None;
    };
    let mut st = store.borrow_mut();
    match st.slots.get_mut(*idx)? {
        Some(Value::Int(i)) => incdec_int_slot(i, kind),
        _ => None,
    }
}

/// See `RW_UNDEF_WATCH`.
#[cold]
fn watch_rw_container(arr: &ZArray) {
    RW_UNDEF_WATCH.with(|w| w.set(Some(arr.clone())));
}

/// After a read-modify-write walk's diagnostics: whether the error handler
/// wrote to or freed the array whose missing key the walk added
/// (`zend_undefined_index_write` / `zend_undefined_offset_write` then
/// return NULL and the operation is skipped; bug70662, bug78598).
fn rw_container_changed(wop: &WOp) -> bool {
    if !matches!(wop, WOp::FetchRw | WOp::AssignOp(..)) {
        return false;
    }
    let changed = RW_UNDEF_WATCH
        .with(std::cell::Cell::take)
        .is_some_and(|a| a.refcount() == 1);
    if changed && matches!(wop, WOp::FetchRw) {
        RW_SKIPPED.with(|c| c.set(true));
    }
    changed
}

/// Performs deferred reference assignments; called once a write walk has
/// dropped its borrows.
pub(crate) fn flush_deferred_ref_assigns() {
    let pending = DEFERRED_REF_ASSIGNS.with(|d| std::mem::take(&mut *d.borrow_mut()));
    for (r, v) in pending {
        let old = r.set(v);
        drop(old);
    }
}

/// Whether `op` on `cur` is the `.=` php-src's typed paths make on the
/// string in place (`zend_binary_assign_op_typed_prop()` /
/// `_typed_ref()`: `concat_function(zptr, zptr, value)`), which a failure
/// leaves as it was, rather than into a temporary that is checked.
fn concats_in_place(op: BinaryOp, cur: &Value) -> bool {
    op == BinaryOp::Concat && matches!(cur, Value::Str(_))
}

pub(crate) fn apply_final(v: &mut Value, wop: &mut WOp) -> Step {
    match wop {
        WOp::Assign(val) => {
            let val = std::mem::take(val);
            match v {
                Value::Reference(r) if crate::typed_refs::has_type_sources(r) => {
                    return Step::TypedRefAssign(r.clone(), val);
                }
                Value::Reference(r) => {
                    // The reference may be the container being walked
                    // (`$a['k'] = null` where `$a['k']` is `&$a`): its value
                    // is replaced once the walk has released it.
                    if r.try_borrow_mut().is_none() {
                        DEFERRED_REF_ASSIGNS
                            .with(|d| d.borrow_mut().push((r.clone(), val.clone())));
                        return Step::Done(val);
                    }
                    let old = r.set(val.clone());
                    crate::gc::possible_root(&old);
                    drop(old);
                }
                _ => {
                    // `zval_ptr_dtor` of the value replaced (an element, a
                    // static property): a possible root when it stays
                    // referenced, as when a variable holding it is
                    // overwritten.
                    let old = std::mem::replace(v, val.clone());
                    crate::gc::possible_root(&old);
                    drop(old);
                }
            }
            Step::Done(val)
        }
        WOp::FetchRw | WOp::AssignOp(..) => Step::Done(v.deref()),
        WOp::MakeRef => Step::Done(Value::Reference(v.make_ref())),
        WOp::AssignRef(r) => {
            let r = std::mem::take(r);
            let old = std::mem::replace(v, r.clone());
            crate::gc::possible_root(&old);
            drop(old);
            Step::Done(r)
        }
        WOp::Unset => {
            let old = std::mem::take(v);
            crate::gc::possible_root(&old);
            drop(old);
            Step::Done(Value::Null)
        }
        WOp::AssignDeref(d) => {
            if let Some(cur) = d.src.try_borrow() {
                // Not a container of this walk: read it now, after the
                // containers were fetched for writing.
                let val = cur.clone();
                drop(cur);
                return apply_final(v, &mut WOp::Assign(val));
            }
            match v {
                Value::Reference(r) => d.hit = DeferredHit::Ref(r.clone()),
                _ => {
                    let old = std::mem::take(v);
                    drop(old);
                    d.hit = DeferredHit::Slot;
                }
            }
            Step::Done(Value::Null)
        }
    }
}

/// `.=` on a slot holding a string, in place. php-src's `ZEND_ASSIGN_DIM_OP`
/// and `ZEND_ASSIGN_STATIC_PROP_OP` operate on the fetched slot
/// (`zend_binary_op(var_ptr, var_ptr, value)`: `concat_function` with
/// `result == op1` extends the string with `zend_string_extend` when
/// nothing else holds it; a shared or interned string is copied once and
/// grows in place from then on), where the generic walk yields a copy of
/// the value, concatenates into a new string of the full length and writes
/// that back with a second, silent walk. `None` leaves the operation to
/// that walk: a right-hand side that is neither a string nor an int (its
/// conversion may warn or call `__toString()`), a slot not holding a
/// string directly (a reference, another type), or a `[]` segment in
/// `segs` (its appended element is recorded for the write-back walk).
/// Callers use it only while no diagnostic of the walk is pending: an
/// error handler raised afterwards could otherwise observe or abort an
/// operation that has already happened. The operation becomes
/// [`WOp::FetchRw`], whose finished step is the result, so the callers'
/// write-back of a compound assignment is skipped.
fn concat_in_place(slot: &mut Value, segs: &[RSeg], wop: &mut WOp) -> Option<Step> {
    let WOp::AssignOp(BinaryOp::Concat, rhs) = wop else {
        return None;
    };
    if segs.iter().any(|seg| matches!(seg, RSeg::Append)) {
        return None;
    }
    let Value::Str(s) = slot else {
        return None;
    };
    match rhs {
        Value::Str(add) => s.append_str(add),
        Value::Int(i) => s.append_long(*i),
        _ => return None,
    }
    let result = Value::Str(s.clone());
    *wop = WOp::FetchRw;
    Some(Step::Done(result))
}

/// The index of the segment whose container the walk over `segs` (from
/// `v`, the container of segment `i`) reached last through `src`, or
/// `None` when the path does not run through it. Used after a
/// [`WOp::AssignDeref`] walk, whose containers are unshared arrays.
fn last_entry(v: &Value, segs: &[RSeg], i: usize, src: &PhpRef) -> Option<usize> {
    if i >= segs.len() {
        return None;
    }
    match v {
        Value::Reference(r) => {
            let deeper = r
                .try_borrow()
                .and_then(|inner| last_entry(&inner, segs, i, src));
            deeper.or_else(|| PhpRef::ptr_eq(r, src).then_some(i))
        }
        Value::Array(a) => {
            // A `[]` segment appended the table's last element.
            let key = match &segs[i] {
                RSeg::Append => a.last().map(|(k, _)| k.clone())?,
                RSeg::Dim(k) => key_of(k, &mut Vec::new(), KeyCtx::Write).ok()?,
                RSeg::Prop(_) => return None,
            };
            last_entry(a.get(&key)?, segs, i + 1, src)
        }
        _ => None,
    }
}

/// Completes a [`DeferredHit::Slot`]: stores a handle to `src`'s array in
/// the final slot, which `segs` (the path from that array, the walk's
/// last entry into `src`) leads to: `$t = &$a; $a[] = $t;` makes `$a`
/// contain itself, like php-src. Returns the assigned value.
fn store_self_reference(segs: &[RSeg], src: &PhpRef) -> Option<Value> {
    let mut path = Vec::with_capacity(segs.len());
    for seg in segs {
        path.push(match seg {
            RSeg::Append => None,
            RSeg::Dim(k) => Some(key_of(k, &mut Vec::new(), KeyCtx::Write).ok()?),
            RSeg::Prop(_) => return None,
        });
    }
    let mut b = src.try_borrow_mut()?;
    let stored = match &mut *b {
        Value::Array(a) => {
            let stored = a.store_self_at(&path);
            if stored.is_some() {
                // The collector tracks arrays that contain themselves, so
                // the cycle is freed like php-src's array roots.
                crate::gc::self_array_created(a);
            }
            stored
        }
        _ => None,
    };
    drop(b);
    stored
}

/// Completes a [`WOp::AssignDeref`] walk over `segs` from variable slot
/// `idx` of `store` that returned `done`, once its borrows are released.
fn finish_deferred(
    done: Value,
    d: &mut DeferredRead,
    store: &Rc<RefCell<VarStore>>,
    idx: usize,
    segs: &[RSeg],
) -> Step {
    match std::mem::replace(&mut d.hit, DeferredHit::None) {
        DeferredHit::None => Step::Done(done),
        DeferredHit::Ref(r) => {
            // The final slot is a reference: assign the source's value
            // through it.
            let v = d.src.try_borrow().map_or(Value::Null, |b| b.clone());
            if crate::typed_refs::has_type_sources(&r) {
                Step::TypedRefAssign(r, v)
            } else {
                let old = r.set(v.clone());
                drop(old);
                crate::gc::release_duplicate(Value::Reference(r));
                Step::Done(v)
            }
        }
        DeferredHit::Slot => {
            let entry = {
                let st = store.borrow();
                st.slots
                    .get(idx)
                    .and_then(Option::as_ref)
                    .and_then(|b| last_entry(b, segs, 0, &d.src))
            };
            let v = entry.and_then(|e| store_self_reference(&segs[e..], &d.src));
            Step::Done(v.unwrap_or(Value::Null))
        }
    }
}

/// How a read-modify-write of a missing dynamic property goes on when the
/// error handler throws at its warning (see [`Vm::rw_dynamic_threw`]).
#[derive(Clone, Copy)]
enum RwThrow<'a> {
    /// `++`/`--`.
    IncDec,
    /// A compound assignment with this operator and right-hand side.
    Op(BinaryOp, &'a Value),
}

impl Vm {
    /// The frame whose symbol table a generic access to place `pid`
    /// attaches only for its own duration: the place starts at a compiled
    /// variable of the running frame, whose table is not attached
    /// ([`Vm::resolve_base`] walks the table's slot). The frame's index and
    /// its table identify it afterwards.
    #[inline]
    fn cv_symtab_frame(&self, pid: PlaceId) -> Option<(usize, *const RefCell<VarStore>)> {
        let f = self.frames.last()?;
        if f.symtab_attached
            || !matches!(
                f.kind,
                crate::vm::FrameKind::Main | crate::vm::FrameKind::Function
            )
        {
            return None;
        }
        let p = f.code.places.get(pid as usize)?;
        matches!(p.base, PlaceBase::Cv(_)).then_some((self.frames.len() - 1, Rc::as_ptr(&f.vars)))
    }

    /// The base of a read (`place_fetch`, `place_isset`) of compiled
    /// variable `cv` in a frame whose symbol table is not attached: the CV
    /// slot is the variable (php-src's `EX_VAR`), so the read takes its
    /// value (dereferenced, as [`Vm::fetch_base_value`] would) or, when it
    /// is undefined, its name, and needs no table. `None` for any other
    /// base or an attached table ([`Vm::resolve_base`] then).
    #[inline]
    fn cv_read_base(&self, base: &PlaceBase) -> Option<RBase> {
        let PlaceBase::Cv(cv) = base else {
            return None;
        };
        let f = self.frame();
        if f.symtab_attached {
            return None;
        }
        let cv = *cv as usize;
        Some(match f.cvs.get(cv).and_then(crate::vm::CvSlot::get) {
            Some(v) => RBase::Value(v.deref()),
            None => RBase::Undef(f.code.cv_names.get(cv)?.clone()),
        })
    }

    /// Runs the generic access `op` to place `pid`, then detaches the
    /// symbol table it attached to reach a compiled variable
    /// ([`Vm::cv_symtab_frame`]): a frame's table stays attached only when
    /// the script asked for it (`$$name`, `extract()`, `global`,
    /// `$GLOBALS`, an include, ...), so the compiled-variable fast paths
    /// keep owning their values (refcount 1, nothing to mirror). php-src
    /// never needs a table for `$cv[...]`; its CV slots are the variables.
    ///
    /// In a function, a defined variable is walked from a private one-slot
    /// table instead ([`CvRoot`]): no copy of the function's other
    /// variables into its table and back for each `$a[$k] = $v`.
    #[inline]
    fn with_cv_symtab<R>(&mut self, pid: PlaceId, op: impl FnOnce(&mut Self) -> R) -> R {
        let Some((fidx, vars)) = self.cv_symtab_frame(pid) else {
            return op(self);
        };
        let root = self.cv_root_of(fidx, pid);
        let pending = root.is_some();
        let prev = if pending {
            std::mem::replace(&mut self.cv_root, root)
        } else {
            None
        };
        let r = op(self);
        if pending {
            if let Some(CvRoot {
                frame,
                cv,
                walk: Some(w),
                ..
            }) = std::mem::replace(&mut self.cv_root, prev)
            {
                self.leave_cv_walk(frame, cv, w);
            }
        }
        if let Some(f) = self.frames.get_mut(fidx) {
            // The same frame (a fiber switch inside the access could have
            // moved the stack): otherwise the table just stays attached.
            if std::ptr::eq(Rc::as_ptr(&f.vars), vars)
                && matches!(
                    f.kind,
                    crate::vm::FrameKind::Main | crate::vm::FrameKind::Function
                )
            {
                f.detach_symbol_table();
            }
        }
        r
    }

    /// Whether a write walk from `base` runs on the private table of a
    /// plain local variable ([`CvWalk`]): it changes no symbol table, so
    /// no frame needs synchronising afterwards.
    #[inline]
    fn cv_walk_private(&self, base: &RBase) -> bool {
        matches!(base, RBase::Var(s, _)
            if self.cv_root.as_ref().and_then(|r| r.walk.as_ref())
                .is_some_and(|w| w.wrapped && Rc::ptr_eq(&w.store, s)))
    }

    /// Whether `store` is the global symbol table (`$GLOBALS`, the global
    /// scope's variables): a write to its slots can leave the compiled
    /// variables of the global-scope frames out of date. A function's own
    /// table, its variables and objects are not in it; a `global $x`
    /// binding brings those frames up to date when it is made
    /// ([`Vm::global_ref`]), and writes through that reference reach the
    /// value the frames share.
    fn store_in_globals(&self, store: &Rc<RefCell<VarStore>>) -> bool {
        Rc::ptr_eq(store, &self.globals)
            || self
                .frames
                .first()
                .is_some_and(|f| Rc::ptr_eq(&f.vars, store))
    }

    /// Drops the handles that compiled variables keep on slot `idx` of
    /// `store` before a walk changes that slot in place: the running
    /// frame's, and, when `store` is the global table, those of every frame
    /// of the global scope attached to it (the main script and top-level
    /// `include`/`eval` frames, [`Vm::sync_global_frames`]). Their copies
    /// would otherwise hold a second handle on the slot's array, and every
    /// write through `$GLOBALS['g'][...]` from a function separated it: the
    /// whole array copied per write (php-src's table slots are
    /// `IS_INDIRECT` into the CVs, `ZEND_FETCH_W` with `ZEND_FETCH_GLOBAL`
    /// writes the one value in place). The sync after the write
    /// ([`Frame::sync_from_symbol_table`], [`Vm::sync_global_frames`])
    /// gives them the new value back.
    ///
    /// [`Frame::sync_from_symbol_table`]: crate::vm::Frame::sync_from_symbol_table
    pub(crate) fn clear_cvs_for_slot(&mut self, store: &Rc<RefCell<VarStore>>, idx: usize) {
        self.frame_mut().clear_cv_for_symtab_slot(store, idx);
        if !Rc::ptr_eq(store, &self.globals) {
            if let Some(main) = self.frames.get_mut(0) {
                if main.symtab_attached {
                    main.clear_cv_for_symtab_slot(store, idx);
                }
            }
            return;
        }
        let running = self.frames.len().saturating_sub(1);
        for f in self
            .frames
            .iter_mut()
            .take(running)
            .take_while(|f| f.kind == crate::vm::FrameKind::Main || Rc::ptr_eq(&f.vars, store))
        {
            if f.symtab_attached {
                f.clear_cv_for_symtab_slot(store, idx);
            }
        }
    }

    /// [`Vm::store_in_globals`] for the base of a write walk
    /// (`$GLOBALS[...]` writes the global table too).
    fn base_in_globals(&self, base: &RBase) -> bool {
        match base {
            RBase::Var(s, _) => self.store_in_globals(s),
            RBase::Globals => true,
            _ => false,
        }
    }

    /// The [`CvRoot`] of a generic write to place `pid` of frame `fidx`
    /// (whose table is not attached): the frame is a function's and the
    /// place starts at a compiled variable. A write to the variable itself
    /// (`++$n`, `unset($a)`, `$a =& $b`) takes it only when the variable
    /// already is a reference (a `static` or `global` one): a plain value
    /// it releases must reach the GC and the destructors as itself, not
    /// wrapped in a reference.
    fn cv_root_of(&self, fidx: usize, pid: PlaceId) -> Option<CvRoot> {
        let f = self.frames.get(fidx)?;
        if f.kind != crate::vm::FrameKind::Function {
            return None;
        }
        let place = f.code.places.get(pid as usize)?;
        let PlaceBase::Cv(cv) = &place.base else {
            return None;
        };
        let wrap = !place.segs.is_empty();
        if !wrap
            && !matches!(
                f.cvs.get(*cv as usize).and_then(crate::vm::CvSlot::get),
                Some(Value::Reference(_))
            )
        {
            return None;
        }
        Some(CvRoot {
            frame: fidx,
            cv: *cv,
            wrap,
            walk: None,
        })
    }

    /// The private table [`Vm::resolve_base`] walks compiled variable `cv`
    /// of the running frame from, when a [`CvRoot`] covers it: made on the
    /// first call, from a variable holding an array, an object or a
    /// reference. `None` sends the walk to the symbol table (an undefined
    /// variable, `false`, a scalar: their messages and conversions are the
    /// table walk's).
    fn cv_root_store(&mut self, cv: u32) -> Option<Rc<RefCell<VarStore>>> {
        let root = self.cv_root.as_ref()?;
        if root.cv != cv || root.frame + 1 != self.frames.len() {
            return None;
        }
        if let Some(w) = &root.walk {
            return Some(w.store.clone());
        }
        let w = self.enter_cv_walk(root.frame, cv, root.wrap)?;
        let store = w.store.clone();
        if let Some(root) = self.cv_root.as_mut() {
            root.walk = Some(w);
        }
        Some(store)
    }

    /// Starts a [`CvWalk`] of compiled variable `cv` of frame `fidx`.
    fn enter_cv_walk(&mut self, fidx: usize, cv: u32, wrap: bool) -> Option<CvWalk> {
        let f = self.frames.get_mut(fidx)?;
        if f.symtab_attached {
            return None;
        }
        let code = f.code.clone();
        let vars = Rc::as_ptr(&f.vars);
        let slot = f.cvs.get_mut(cv as usize)?;
        let (r, wrapped) = match slot.get()? {
            Value::Reference(r) => (r.clone(), false),
            Value::Array(_) | Value::Object(_) if wrap => {
                let r = PhpRef::new(slot.set(None)?);
                slot.set(Some(Value::Reference(r.clone())));
                (r, true)
            }
            _ => return None,
        };
        // Kept for the next walk: like php-src's preallocated tables, not
        // part of `memory_get_usage()`.
        let store = match self.cv_scratch.take() {
            Some(s) if Rc::strong_count(&s) == 1 => s,
            _ => php_memlimit::persistent(|| Rc::new(RefCell::new(VarStore::default()))),
        };
        php_memlimit::persistent(|| store.borrow_mut().reset_for(&code));
        if let Some(s) = store.borrow_mut().slots.get_mut(cv as usize) {
            *s = Some(Value::Reference(r.clone()));
        }
        Some(CvWalk {
            vars,
            store,
            r,
            wrapped,
        })
    }

    /// Ends a [`CvWalk`]: the variable takes what the private slot holds
    /// now (another reference after `=&`, nothing after `unset()`), and a
    /// reference made for the walk that nothing else took becomes the
    /// plain value again. The value leaves the reference before any handle
    /// of it is dropped, so no handle release records it as a possible
    /// cycle root.
    fn leave_cv_walk(&mut self, frame: usize, cv: u32, w: CvWalk) {
        let CvWalk {
            vars,
            store,
            r,
            wrapped,
        } = w;
        let after = store
            .borrow_mut()
            .slots
            .get_mut(cv as usize)
            .and_then(Option::take);
        // Kept for the next walk (or let go) outside `memory_get_usage()`,
        // as it was made.
        let keep = &mut self.cv_scratch;
        php_memlimit::persistent(move || {
            if Rc::strong_count(&store) == 1 {
                *keep = Some(store);
            }
        });
        // The same frame: a fiber switch inside the walk could have moved
        // the stack (the variable then just stays a reference).
        let Some(f) = self
            .frames
            .get_mut(frame)
            .filter(|f| std::ptr::eq(Rc::as_ptr(&f.vars), vars))
        else {
            return;
        };
        let Some(slot) = f.cvs.get_mut(cv as usize) else {
            return;
        };
        let new = match after {
            Some(Value::Reference(now)) if PhpRef::ptr_eq(&now, &r) => {
                // Held by the private slot, the variable and `r` only.
                let ours = matches!(slot.get(), Some(Value::Reference(c)) if PhpRef::ptr_eq(c, &r));
                if !wrapped || !ours || r.refcount() != 3 {
                    return;
                }
                let v = r.set(Value::Null);
                drop(now);
                Some(v)
            }
            other => other,
        };
        let old = slot.set(new);
        drop(old);
    }

    /// `place = $cv` where `$cv` may hold a reference ([`Vm::place_assign_cv_ref_impl`]).
    pub(crate) fn place_assign_cv_ref(&mut self, pid: PlaceId, cv: u32) -> VmResult<Value> {
        self.with_cv_symtab(pid, |vm| vm.place_assign_cv_ref_impl(pid, cv))
    }

    /// `place = v`.
    pub(crate) fn place_assign(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        // `$this->p[$k][$j] = v` & co. into an array property: no walk.
        let v = match self.assign_prop_dims_fast(pid, v) {
            Ok(r) => return Ok(r),
            Err(v) => v,
        };
        self.vivify_cv_array(pid);
        self.with_cv_symtab(pid, |vm| vm.place_assign_impl(pid, v))
    }

    /// `$cv[...] = v` / `$cv[] = v` on an undefined or null variable of a
    /// function whose symbol table is not attached: `ZEND_ASSIGN_DIM` and
    /// `ZEND_FETCH_DIM_W` make the variable an empty array before they
    /// look at the offset (no diagnostic; the array stays when the offset
    /// then throws). Made here, the walk starts from the array in the
    /// variable's private one-slot table ([`CvRoot`]) instead of from the
    /// whole table attached for an undefined variable (every variable of
    /// the frame copied in and back), which is what `$list[] = $x` without
    /// an initialisation cost each time. A reference (possibly typed) and
    /// `false` (deprecated) keep the generic walk.
    #[inline]
    fn vivify_cv_array(&mut self, pid: PlaceId) {
        let Some(f) = self.frames.last_mut() else {
            return;
        };
        if f.symtab_attached || f.kind != crate::vm::FrameKind::Function {
            return;
        }
        let Some(p) = f.code.places.get(pid as usize) else {
            return;
        };
        let (PlaceBase::Cv(cv), Some(Seg::Dim(_) | Seg::Append)) = (&p.base, p.segs.first()) else {
            return;
        };
        let cv = *cv as usize;
        if let Some(slot) = f.cvs.get_mut(cv) {
            let slot = slot.as_mut();
            if matches!(slot, None | Some(Value::Null)) {
                *slot = Some(Value::Array(ZArray::new()));
            }
        }
    }

    /// `place = $cv` with `$cv` undefined ([`Vm::place_assign_undef_cv_impl`]).
    pub(crate) fn place_assign_undef_cv(&mut self, pid: PlaceId, cv: u32) -> VmResult<Value> {
        self.with_cv_symtab(pid, |vm| vm.place_assign_undef_cv_impl(pid, cv))
    }

    /// `++place` etc.
    pub(crate) fn place_incdec(&mut self, pid: PlaceId, kind: IncDecKind) -> VmResult<Value> {
        self.with_cv_symtab(pid, |vm| vm.place_incdec_impl(pid, kind))
    }

    /// `place =& v`.
    pub(crate) fn place_assign_ref(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        let v = match self.cv_assign_ref_fast(pid, v) {
            Ok(r) => return Ok(r),
            Err(v) => v,
        };
        self.with_cv_symtab(pid, |vm| vm.place_assign_ref_impl(pid, v))
    }

    /// `$cv =& <reference>` (`ZEND_ASSIGN_REF` with a CV result operand,
    /// `zend_assign_to_variable_reference`, `Zend/zend_execute.c`) in a
    /// function whose symbol table is not attached: the CV slot is the
    /// variable, so it is bound to the reference in place and its old value
    /// released (`zval_ptr_dtor`: a possible root, a destructor when it was
    /// the last handle), as [`Vm::write_var_root`] does on a table slot,
    /// minus the attach/detach round trip of [`Vm::with_cv_symtab`] that
    /// copied all the frame's variables into the table and back for each
    /// `$a = &$b` and each iteration of `foreach (... as &$v)`. `Err` hands
    /// `v` back, unchanged, for any other place, frame state or a value
    /// that is not a reference ("Only variables should be assigned by
    /// reference" and the other diagnostics stay the generic path's).
    #[inline]
    fn cv_assign_ref_fast(&mut self, pid: PlaceId, v: Value) -> Result<Value, Value> {
        let Value::Reference(r) = v else {
            return Err(v);
        };
        let Some(f) = self.frames.last_mut() else {
            return Err(Value::Reference(r));
        };
        // Not the main script's frame: the global table may still hold its
        // own handle of the old value ([`Vm::bare_cv_slot`]).
        if f.symtab_attached || f.kind != crate::vm::FrameKind::Function {
            return Err(Value::Reference(r));
        }
        let cv = match f.code.places.get(pid as usize) {
            Some(p) if p.segs.is_empty() => match p.base {
                PlaceBase::Cv(cv) => cv as usize,
                _ => return Err(Value::Reference(r)),
            },
            _ => return Err(Value::Reference(r)),
        };
        let Some(slot) = f.cvs.get_mut(cv) else {
            return Err(Value::Reference(r));
        };
        let old = slot.set(Some(Value::Reference(r.clone())));
        if let Some(old) = &old {
            crate::gc::possible_root(old);
        }
        drop(old);
        Ok(Value::Reference(r))
    }

    /// Makes a place a reference.
    pub(crate) fn place_make_ref(&mut self, pid: PlaceId) -> VmResult<PhpRef> {
        if let Some(r) = self.cv_make_ref_fast(pid) {
            return Ok(r);
        }
        self.with_cv_symtab(pid, |vm| vm.place_make_ref_impl(pid))
    }

    /// `&$cv` (`ZEND_SEND_REF`, `ZEND_MAKE_REF` of a plain compiled
    /// variable) in a frame whose symbol table is not attached: the CV slot
    /// is the variable, so it becomes (or already is) a reference in place,
    /// an undefined one a reference to null without a notice. This is what
    /// the general path does after attaching the table
    /// ([`Vm::with_cv_symtab`] and `write_var_root`'s `WOp::MakeRef`), minus
    /// the attach/detach round trip that made every by-reference argument
    /// (`preg_match($re, $s, $m)`) cost a copy of all the frame's variables.
    /// `None` for any other place or frame state.
    #[inline]
    pub(crate) fn cv_make_ref_fast(&mut self, pid: PlaceId) -> Option<PhpRef> {
        let f = self.frames.last_mut()?;
        if f.symtab_attached
            || !matches!(
                f.kind,
                crate::vm::FrameKind::Main | crate::vm::FrameKind::Function
            )
        {
            return None;
        }
        let p = f.code.places.get(pid as usize)?;
        let (PlaceBase::Cv(cv), true) = (&p.base, p.segs.is_empty()) else {
            return None;
        };
        let cv = *cv as usize;
        let slot = f.cvs.get_mut(cv)?.as_mut();
        Some(slot.get_or_insert(Value::Null).make_ref())
    }

    /// The reference a by-reference argument of a prepared native call
    /// (`calls::send_ref`) gets for place `pid` of the current frame, when
    /// making it can neither report anything nor run user code, so that no
    /// frame is pushed over the prepared one: `$this->p` or `$o->p` on a
    /// property cache hit ([`Vm::prop_cache_make_ref`], as
    /// [`Vm::place_make_ref`] starts), or `$a[$k]` / `$this->p[$k]` of an
    /// array held by a compiled variable (directly or through a reference)
    /// or by a cached declared property ([`Vm::prop_cache_container`]),
    /// `$k` an int or string literal or compiled variable: the element
    /// becomes a reference, created as null when missing
    /// (`ZEND_FETCH_DIM_W`, no notice) and the array separated first when
    /// shared (`SEPARATE_ARRAY`). `None`, having changed nothing, for any
    /// other place or frame state (the generic send then makes it).
    pub(crate) fn place_make_ref_quiet(&mut self, pid: PlaceId) -> Option<PhpRef> {
        /// What the place is, read before anything changes.
        enum Shape {
            Prop,
            CvDim(u32, ArrayKey),
            PropDim(ArrayKey),
        }
        let shape = {
            let f = self.frames.last()?;
            if f.symtab_attached
                || !matches!(
                    f.kind,
                    crate::vm::FrameKind::Main | crate::vm::FrameKind::Function
                )
            {
                return None;
            }
            let p = f.code.places.get(pid as usize)?;
            match (&p.base, p.segs.as_slice()) {
                (PlaceBase::Cv(_) | PlaceBase::This, [Seg::Prop(Operand::Lit(_))]) => Shape::Prop,
                (PlaceBase::Cv(cv), [Seg::Dim(k)]) => Shape::CvDim(*cv, quiet_key(f, *k)?),
                (PlaceBase::Cv(_) | PlaceBase::This, [Seg::Prop(Operand::Lit(_)), Seg::Dim(k)]) => {
                    Shape::PropDim(quiet_key(f, *k)?)
                }
                _ => return None,
            }
        };
        match shape {
            Shape::Prop => self.prop_cache_make_ref(pid),
            Shape::CvDim(cv, key) => {
                let slot = self.frames.last_mut()?.cvs.get_mut(cv as usize)?;
                match slot.as_mut().as_mut()? {
                    Value::Array(a) => Some(element_make_ref(a, key)),
                    Value::Reference(r) => {
                        let mut held = r.try_borrow_mut()?;
                        match &mut *held {
                            Value::Array(a) => Some(element_make_ref(a, key)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            Shape::PropDim(key) => {
                let (obj, idx) = self.prop_cache_container(pid)?;
                let data = object::data(&obj);
                let mut slots = data.slots.try_borrow_mut().ok()?;
                match slots.get_mut(idx)?.as_mut()? {
                    Value::Array(a) => Some(element_make_ref(a, key)),
                    _ => None,
                }
            }
        }
    }

    /// `unset(place)`.
    pub(crate) fn place_unset(&mut self, pid: PlaceId) -> VmResult<()> {
        // `unset($a[$k])`, `unset($this->p[$k][$j])` & co. on arrays: no
        // walk, no symbol table.
        if self.unset_dims_fast(pid) {
            return Ok(());
        }
        if self.cv_unset_fast(pid) {
            return Ok(());
        }
        self.with_cv_symtab(pid, |vm| vm.place_unset_impl(pid))
    }

    /// The bare compiled variable `place` names in the running frame, when
    /// that frame is a function's and its symbol table is not attached:
    /// the CV slot is then the variable (php-src's `EX_VAR`), and an access
    /// to it needs no table. Not the main script's: the global table keeps
    /// its own entry for each of its variables, which an undefined slot
    /// would not clear when the table is attached again
    /// ([`crate::vm::Frame::sync_cvs_to_symtab`] skips undefined slots).
    #[inline]
    fn bare_cv_slot(&self, pid: PlaceId) -> Option<u32> {
        let f = self.frames.last()?;
        if f.symtab_attached || f.kind != crate::vm::FrameKind::Function {
            return None;
        }
        let p = f.code.places.get(pid as usize)?;
        match (&p.base, p.segs.is_empty()) {
            (PlaceBase::Cv(cv), true) => Some(*cv),
            _ => None,
        }
    }

    /// `unset($cv)` (`ZEND_UNSET_CV`) in a frame whose symbol table is not
    /// attached: the slot becomes undefined, then the old value is
    /// released (a possible cycle root; its destructor, which already sees
    /// the variable unset, when this was its last handle), as
    /// [`Vm::write_var_root`] does on a table slot, without
    /// [`Vm::with_cv_symtab`] copying all the frame's variables into the
    /// table and back. `false`, having changed nothing, for any other
    /// place or frame state.
    #[inline]
    fn cv_unset_fast(&mut self, pid: PlaceId) -> bool {
        let Some(cv) = self.bare_cv_slot(pid) else {
            return false;
        };
        let Some(slot) = self
            .frames
            .last_mut()
            .and_then(|f| f.cvs.get_mut(cv as usize))
        else {
            return false;
        };
        let old = slot.set(None);
        if let Some(old) = &old {
            crate::gc::possible_root(old);
        }
        drop(old);
        true
    }

    /// `place op= v`.
    pub(crate) fn place_assign_op(
        &mut self,
        pid: PlaceId,
        op: BinaryOp,
        v: Value,
    ) -> VmResult<Value> {
        if let Some(r) = self.assign_op_cv_scalar(pid, op, &v) {
            return r;
        }
        let v = match self.assign_op_cv_slot(pid, op, v) {
            Ok(r) => return r,
            Err(v) => v,
        };
        self.with_cv_symtab(pid, |vm| vm.place_assign_op_impl(pid, op, v, None))
    }

    /// `$cv op= v` (`ZEND_ASSIGN_OP` on a CV) of a defined variable of a
    /// function whose symbol table is not attached, for the operands
    /// [`Vm::assign_op_cv_scalar`] leaves out: `$s .= $object` (a
    /// `__toString()` call), an array, an operand whose conversion warns.
    /// The generic path's steps on the slot itself: the current value
    /// (through a reference, if the variable is one) is read, the
    /// operation runs (its diagnostics and user code with it), and the
    /// result is written as [`Vm::write_var_root`] writes it (through a
    /// typed reference's checks; the old value released as a possible
    /// root). Without [`Vm::with_cv_symtab`], no copy of all the frame's
    /// variables into the table and back around the operation (about 2,600
    /// of the 7,000 instructions of `$s .= $object`). `Err` hands `v` back,
    /// having changed nothing, for an undefined variable (whose warning
    /// and exception order the generic path keeps), `$array += $array`
    /// (made in place there), any other place or frame state.
    fn assign_op_cv_slot(
        &mut self,
        pid: PlaceId,
        op: BinaryOp,
        v: Value,
    ) -> Result<VmResult<Value>, Value> {
        let Some(cv) = self.bare_cv_slot(pid) else {
            return Err(v);
        };
        let Some(f) = self.frames.last() else {
            return Err(v);
        };
        let vars = Rc::as_ptr(&f.vars);
        let cur = match f.cvs.get(cv as usize).and_then(crate::vm::CvSlot::get) {
            Some(cur) => cur.deref(),
            None => return Err(v),
        };
        if op == BinaryOp::Add && matches!((&cur, &v), (Value::Array(_), Value::Array(_))) {
            return Err(v);
        }
        let nv = match self.assign_op_value(op, &cur, &v) {
            Ok(nv) => nv,
            Err(t) if concats_in_place(op, &cur) => return Ok(Err(t)),
            Err(t) => {
                drop(cur);
                return Ok(Err(self.failed_assign_op_cv(cv, vars, t)));
            }
        };
        // The old value's handle is released before the write, so that
        // the release below sees the count php-src's would.
        drop(cur);
        // The same frame: user code run by the operation (`__toString()`)
        // returned to it, unless a fiber switched stacks meanwhile.
        let Some(f) = self
            .frames
            .last_mut()
            .filter(|f| std::ptr::eq(Rc::as_ptr(&f.vars), vars))
        else {
            return Ok(Ok(nv));
        };
        let Some(slot) = f.cvs.get_mut(cv as usize) else {
            return Ok(Ok(nv));
        };
        let held = match slot.get() {
            Some(Value::Reference(r)) => Some(r.clone()),
            _ => None,
        };
        let old = match held {
            Some(r) if crate::typed_refs::has_type_sources(&r) => {
                return Ok(self.assign_ref(&r, nv));
            }
            // The table, if user code attached it meanwhile, holds the
            // same reference.
            Some(r) => Some(r.set(nv.clone())),
            None => {
                let old = slot.set(Some(nv.clone()));
                // Attached by user code meanwhile: the table mirrors the slot.
                f.sync_cv_to_symtab(cv, &nv);
                old
            }
        };
        if let Some(old) = &old {
            crate::gc::possible_root(old);
        }
        drop(old);
        Ok(Ok(nv))
    }

    /// `$cv op= v` (`ZEND_ASSIGN_OP` on a compiled variable) without the
    /// symbol table and the place walk, when nothing but its result can
    /// come of it: the variable, of a frame whose symbol table is not
    /// attached, holds an int or a float, and `v` is null, a bool, an int,
    /// a float or, for `+`, `-`, `*` and `/`, a numeric string without
    /// trailing data, whose juggling reports nothing; `%`, the shifts and
    /// the bitwise operators take ints, bools and null only (a fractional
    /// float's implicit conversion to int is deprecated), and `**` is left
    /// out (`0 ** -1` is deprecated). The operation is the generic path's
    /// (`ops::binary_op`, then the result in place of the old number,
    /// which needs no release): division or modulo by zero and a negative
    /// shift throw with the variable unchanged. `None`, having changed
    /// nothing, for every other case.
    fn assign_op_cv_scalar(
        &mut self,
        pid: PlaceId,
        op: BinaryOp,
        v: &Value,
    ) -> Option<VmResult<Value>> {
        use BinaryOp::{Add, BitAnd, BitOr, BitXor, Div, Mod, Mul, ShiftLeft, ShiftRight, Sub};
        let f = self.frames.last()?;
        if f.symtab_attached {
            return None;
        }
        let p = f.code.places.get(pid as usize)?;
        let (PlaceBase::Cv(cv), []) = (&p.base, p.segs.as_slice()) else {
            return None;
        };
        let cv = *cv;
        let cur = match f.cvs.get(cv as usize).and_then(crate::vm::CvSlot::get)? {
            n @ (Value::Int(_) | Value::Float(_)) => n.clone(),
            _ => return None,
        };
        let silent = match (op, v) {
            (
                Add | Sub | Mul | Div,
                Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_),
            ) => true,
            (Add | Sub | Mul | Div, Value::Str(s)) => {
                php_runtime::numeric::is_numeric(s.as_bytes())
            }
            (
                Mod | ShiftLeft | ShiftRight | BitOr | BitAnd | BitXor,
                Value::Null | Value::Bool(_) | Value::Int(_),
            ) => matches!(cur, Value::Int(_)),
            _ => false,
        };
        if !silent {
            return None;
        }
        let nv = match ops::binary_op(op, &cur, v, self) {
            Ok(nv) => nv,
            Err(e) => return Some(Err(self.rt_throw(e))),
        };
        if let Some(slot) = self
            .frames
            .last_mut()
            .and_then(|f| f.cvs.get_mut(cv as usize))
            .map(crate::vm::CvSlot::as_mut)
        {
            // The old value is an int or a float: nothing to release.
            *slot = Some(nv.clone());
        }
        Some(Ok(nv))
    }

    /// `place op= $cv` with `$cv` undefined and `place` ending in a
    /// dimension: `ZEND_ASSIGN_DIM_OP` reads `OP_DATA` once it has the
    /// slot (after the container's and the offset's diagnostics), an
    /// object's `zend_binary_assign_op_obj_dim` before its handlers, and a
    /// scalar or string container throws without reading it.
    pub(crate) fn place_assign_op_undef_cv(
        &mut self,
        pid: PlaceId,
        op: BinaryOp,
        cv: u32,
    ) -> VmResult<Value> {
        self.with_cv_symtab(pid, |vm| {
            vm.place_assign_op_impl(pid, op, Value::Null, Some(cv))
        })
    }

    /// Whether `place op= v` reads `v` during the walk when `v` is an
    /// undefined compiled variable ([`Vm::place_assign_op_undef_cv`]).
    pub(crate) fn assign_op_reads_value_late(&self, pid: PlaceId) -> bool {
        matches!(
            self.frame().code.places[pid as usize].segs.last(),
            Some(Seg::Dim(_) | Seg::Append)
        )
    }

    /// Resolves a place's base and segments.
    #[inline]
    pub(crate) fn resolve_place(&mut self, pid: PlaceId, write: bool) -> VmResult<(RBase, Segs)> {
        self.resolve_place_ex(pid, write, false, PlaceOp::Other)
    }

    /// [`Vm::resolve_place`]; `quiet` is the `BP_VAR_IS` fetch of
    /// `isset()`/`empty()`, where an undeclared or inaccessible static
    /// property base resolves to an undefined base without an error.
    ///
    /// For a write, an offset that is an undefined compiled variable
    /// resolves to null without the warning, which the walk raises when it
    /// gets to the offset ([`PendingRead`]).
    ///
    /// `op` tells a static property base how it is written (see
    /// [`StaticWrite`]).
    fn resolve_place_ex(
        &mut self,
        pid: PlaceId,
        write: bool,
        quiet: bool,
        op: PlaceOp,
    ) -> VmResult<(RBase, Segs)> {
        let code = self.frame().code.clone();
        let p = &code.places[pid as usize];
        let sw = match (write, p.segs.is_empty(), op) {
            (true, true, PlaceOp::Assign) => Some(StaticWrite::Assign),
            (true, true, _) => Some(StaticWrite::Indirect),
            (true, false, _) => Some(StaticWrite::Fetch),
            (false, false, PlaceOp::Unset) => Some(StaticWrite::FetchUnset),
            (false, _, _) => None,
        };
        let base = self.resolve_base(&p.base, write, quiet, sw)?;
        let n = p.segs.len();
        let mut segs = Vec::with_capacity(n);
        let mut undefined = Vec::new();
        for (j, s) in p.segs.iter().enumerate() {
            segs.push(match s {
                Seg::Dim(Operand::Cv(cv)) if write => match self.read_cv_defined(*cv) {
                    Some(v) => RSeg::Dim(v),
                    None => {
                        undefined.push((j, code.cv_names[*cv as usize].clone()));
                        RSeg::Dim(Value::Null)
                    }
                },
                Seg::Dim(op) => RSeg::Dim(self.read_dim_operand(*op)?),
                Seg::Append => RSeg::Append,
                Seg::Prop(op) => {
                    let v = self.read(*op)?;
                    RSeg::Prop(self.to_zstr(&v)?)
                }
            });
        }
        let mut segs = Segs {
            v: segs,
            reads: false,
        };
        for (j, name) in undefined {
            segs.add_read(j, name);
        }
        Ok((base, segs))
    }

    /// Resolves a place's base: the variable, `$this`, a static property
    /// cell or a temporary (reading the name operands of `$$n`/`A::$n`).
    /// `quiet` is `BP_VAR_IS`: an undeclared or inaccessible static
    /// property base resolves to an undefined base without an error.
    /// `sw` is how a static property base is written, for its set
    /// visibility check.
    fn resolve_base(
        &mut self,
        base: &PlaceBase,
        write: bool,
        quiet: bool,
        sw: Option<StaticWrite>,
    ) -> VmResult<RBase> {
        Ok(match base {
            PlaceBase::Cv(cv) => {
                if self.cv_root.is_some() {
                    if let Some(store) = self.cv_root_store(*cv) {
                        return Ok(RBase::Var(store, *cv as usize));
                    }
                }
                // The walk works on the symbol table's slot: attach it (the
                // entry points undo that afterwards, see
                // [`Vm::with_cv_symtab`]) or bring it up to date.
                let f = self.frame_mut();
                if f.symtab_attached {
                    f.sync_cvs_to_symtab();
                } else {
                    f.attach_symbol_table();
                }
                let f = self.frame();
                let idx = if let Some(name) = f.code.cv_names.get(*cv as usize) {
                    let mut st = f.vars.borrow_mut();
                    st.find_or_add_cv(*cv as usize, name.as_bytes()) as usize
                } else {
                    *cv as usize
                };
                RBase::Var(f.vars.clone(), idx)
            }
            PlaceBase::VarVar(op) => {
                let n = self.read(*op)?;
                let n = self.to_zstr(&n)?;
                if n.as_bytes() == b"this" {
                    // `zend_fetch_var_address_helper`: `$this` is never in
                    // the symbol table, so `zend_fetch_this_var` runs: a
                    // write fetch throws, a read outside an object is an
                    // undefined variable.
                    if write {
                        return Err(self.make_throw("Error", "Cannot re-assign $this"));
                    }
                    match self.frame().this.clone() {
                        Some(o) => RBase::This(Some(o)),
                        None => RBase::Undef(n),
                    }
                } else {
                    self.frame_mut().attach_symbol_table();
                    let vars = self.frame().vars.clone();
                    let idx = if write {
                        Some(vars.borrow_mut().find_or_add(n.as_bytes()))
                    } else {
                        vars.borrow_mut().find(n.as_bytes())
                    };
                    match idx {
                        Some(i) => RBase::Var(vars, i as usize),
                        None => RBase::Undef(n),
                    }
                }
            }
            PlaceBase::Global(op) => {
                self.attach_globals_symbol_table();
                let n = self.read(*op)?;
                let n = self.to_zstr(&n)?;
                let bytes = n.as_bytes();
                let g = self.globals.clone();
                let idx = if let Some(ag_idx) = crate::auto_globals::index(bytes) {
                    if let Some(i) = self.ext.auto_globals.cached_slots[ag_idx] {
                        Some(i)
                    } else {
                        let i = g.borrow_mut().find_or_add(bytes);
                        self.ext.auto_globals.cached_slots[ag_idx] = Some(i);
                        Some(i)
                    }
                } else if write || is_superglobal_name(bytes) {
                    Some(g.borrow_mut().find_or_add(bytes))
                } else {
                    g.borrow_mut().find(bytes)
                };
                match idx {
                    Some(i) => RBase::Var(g, i as usize),
                    None => RBase::Undef(n),
                }
            }
            PlaceBase::GlobalsArray => {
                self.attach_globals_symbol_table();
                RBase::Globals
            }
            PlaceBase::This => RBase::This(self.frame().this.clone()),
            PlaceBase::StaticProp { class, name } => {
                // The op's run-time cache slot (`CACHED_PTR`, see
                // `rtcache`): the class and name are only looked up on a
                // miss.
                if let Some(cell) = self.static_prop_hit(class, *name) {
                    return Ok(RBase::Cell(cell));
                }
                let n = self.read(*name)?;
                let n = self.to_zstr(&n)?;
                let cls = self.resolve_class_spec(class)?;
                let r = if quiet {
                    match self.static_prop_cell_is(&cls, &n)? {
                        Some(cell) => RBase::Cell(cell),
                        None => RBase::Undef(n),
                    }
                } else if let Some(sw) = sw {
                    let (cell, fl) = self.static_prop_cell_flags(&cls, &n)?;
                    if fl & crate::func::flags::SET_MASK != 0
                        && self.check_static_set_access(&cls, &n, &cell, sw)?
                    {
                        RBase::Undef(n)
                    } else {
                        RBase::Cell(cell)
                    }
                } else {
                    RBase::Cell(self.static_prop_cell(&cls, &n)?)
                };
                if matches!(r, RBase::Cell(_)) {
                    self.static_prop_fill(class, *name);
                }
                r
            }
            PlaceBase::Value(op) => {
                let v = self.read_raw(*op)?;
                RBase::Value(v)
            }
        })
    }

    fn read_dim_operand(&mut self, op: Operand) -> VmResult<Value> {
        self.read(op)
    }

    /// Emits deferred diagnostics.
    pub(crate) fn flush_diags(&mut self, diags: Diags) -> VmResult<()> {
        for (level, msg) in diags {
            self.error(level, &msg)?;
        }
        Ok(())
    }

    pub(crate) fn walk_err(&mut self, e: WalkErr) -> Throw {
        match e {
            WalkErr::Error(m) => self.make_throw("Error", &m),
            WalkErr::TypeError(m) => self.make_throw("TypeError", &m),
            // Only reached when no caller could convert the value.
            WalkErr::StringOffsetObject => {
                self.make_throw("Error", "Cannot assign an object to a string offset")
            }
        }
    }

    /// The rest of `zend_assign_to_string_offset` for an object value, once
    /// the walk has checked the offset ([`WalkErr::StringOffsetObject`]):
    /// converts it (`__toString()` may throw, leaving the string as it
    /// was), warns when more than one byte results, and writes the first
    /// byte by walking the place again (`rewalk`) with the converted value.
    #[cold]
    #[inline(never)]
    fn assign_string_offset_object(
        &mut self,
        wop: &WOp,
        rewalk: impl FnOnce(&mut Self, WOp) -> VmResult<Value>,
    ) -> VmResult<Value> {
        let WOp::Assign(val) = wop else {
            return Err(self.walk_err(WalkErr::StringOffsetObject));
        };
        let s = self.to_zstr(&val.deref())?;
        let first = match s.as_bytes() {
            [] => ZStr::empty(),
            [c] => ZStr::from_byte(*c),
            [c, ..] => {
                let c = *c;
                self.warning("Only the first byte will be assigned to the string offset")?;
                ZStr::from_byte(c)
            }
        };
        rewalk(self, WOp::Assign(Value::Str(first)))
    }

    /// Runs a write operation on a place.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn write_place(&mut self, base: RBase, segs: &[RSeg], wop: WOp) -> VmResult<Value> {
        let private = self.cv_walk_private(&base);
        let globals = self.base_in_globals(&base);
        if self.writes_global_table(&base) {
            // The running frame of the global scope (an included file's
            // code) has written its compiled variables without mirroring
            // them in the table (the fast paths): they go there first, as
            // a write to a variable of its own brings them up to date
            // (`resolve_base`), or the sync after the write would take the
            // table's older values back into them.
            let table = Rc::clone(&self.globals);
            let f = self.frame_mut();
            if f.symtab_attached && Rc::ptr_eq(&f.vars, &table) {
                f.sync_cvs_to_symtab();
            }
        }
        let r = self.write_place_inner(base, segs, wop, false);
        flush_deferred_ref_assigns();
        if !private {
            self.frame_mut().sync_from_symbol_table();
            if globals {
                self.sync_global_frames();
            }
        }
        r
    }

    /// Whether a write to `base` goes into the global symbol table, whose
    /// change the frames of the global scope then take into their compiled
    /// variables ([`Vm::sync_global_frames`]): `$GLOBALS[...]`, or a
    /// variable of the global table itself. Any other write (a function's
    /// own variable, `$this`, a static property, a temporary) cannot change
    /// it: a variable bound by `global $x` shares its value with the table
    /// through a reference, which both see. Syncing every frame of the
    /// global scope after each write cost the Laravel app's unit tests
    /// about 2% of their instructions.
    fn writes_global_table(&self, base: &RBase) -> bool {
        match base {
            RBase::Globals => true,
            RBase::Var(store, _) => Rc::ptr_eq(store, &self.globals),
            RBase::This(_) | RBase::Cell(_) | RBase::Value(_) | RBase::Undef(_) => false,
        }
    }

    /// `quiet` drops the walk's deferred diagnostics: the write-back walk
    /// of a compound assignment repeats offset conversions the first walk
    /// already reported. Otherwise, with an error handler set, the base
    /// container's walk runs live ([`Vm::run_live`]).
    #[allow(clippy::too_many_lines)]
    fn write_place_inner(
        &mut self,
        base: RBase,
        segs: &[RSeg],
        mut wop: WOp,
        quiet: bool,
    ) -> VmResult<Value> {
        let again = matches!(
            wop,
            WOp::AssignOp(..) | WOp::FetchRw | WOp::Assign(Value::Object(_) | Value::Reference(_))
        )
        .then(|| base.clone());
        let live = !quiet && self.live_walk(segs, &wop);
        let mut diags = Vec::new();
        let step = match base {
            RBase::Var(store, idx) => {
                if segs.is_empty() {
                    return self.write_var_root(&store, idx, wop);
                }
                let rw = matches!(wop, WOp::FetchRw | WOp::AssignOp(..));
                let mut st = store.borrow_mut();
                let undef = st.slots[idx].is_none();
                if undef {
                    if matches!(wop, WOp::Unset) {
                        return Ok(Value::Null);
                    }
                    if rw {
                        let name = st.names[idx].clone();
                        drop(st);
                        self.warning(&crate::exec::undefined_var_msg(&name))?;
                        st = store.borrow_mut();
                    }
                    if matches!(segs[0], RSeg::Prop(_)) {
                        // `ASSIGN_OBJ`/`FETCH_OBJ_W` on an undefined CV
                        // throw without defining it (`zend_throw_non_object_error`).
                        drop(st);
                        let mut null = Value::Null;
                        let s = descend(&mut null, segs, 0, &mut wop, &mut diags, true);
                        if !quiet {
                            self.flush_diags(diags)?;
                        }
                        return self.finish_step(s, segs, wop);
                    }
                    st.slots[idx] = Some(Value::Null);
                }
                if live {
                    drop(st);
                    // As below: the walk mutates the shared slot in place,
                    // so the frame's copy must not be written back over it
                    // when the error handler attaches or syncs the symbol
                    // table (`global $x` / `$GLOBALS` writes; bug79793).
                    self.clear_cvs_for_slot(&store, idx);
                    let mut walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
                        let mut st = store.borrow_mut();
                        let step = match st.slots.get_mut(idx) {
                            Some(Some(slot)) => descend(slot, sg, 0, w, d, false),
                            _ => Step::Done(Value::Null),
                        };
                        drop(st);
                        step
                    };
                    match self.run_live(segs, &mut wop, &mut walk) {
                        Live::Walked(s, d) => {
                            diags = d;
                            s
                        }
                        Live::Over(r) => return r,
                    }
                } else {
                    drop(st);
                    self.clear_cvs_for_slot(&store, idx);
                    let mut st = store.borrow_mut();
                    let slot = st.slots[idx].as_mut().expect("slot defined");
                    let s = descend(slot, segs, 0, &mut wop, &mut diags, undef);
                    drop(st);
                    s
                }
            }
            RBase::This(o) => match o {
                Some(o) if segs.is_empty() => {
                    return Ok(match wop {
                        WOp::MakeRef => Value::Reference(PhpRef::new(Value::Object(o))),
                        _ => Value::Object(o),
                    });
                }
                Some(o) => Step::Obj(o, 0),
                None => {
                    return Err(self.make_throw("Error", "Using $this when not in object context"))
                }
            },
            RBase::Cell(cell) => {
                if let Some(v) = self.typed_static_write(&cell, segs, &mut wop)? {
                    return Ok(v);
                }
                if segs.is_empty() {
                    let mut b = cell.borrow_mut();
                    // `ZEND_ASSIGN_STATIC_PROP_OP` with `.=` on a string
                    // extends it in place (`typed_static_write` has let a
                    // typed property holding a string through).
                    let s = match concat_in_place(&mut b, segs, &mut wop) {
                        Some(s) => s,
                        None => apply_final(&mut b, &mut wop),
                    };
                    drop(b);
                    return self.finish_step(s, segs, wop);
                }
                if live {
                    let mut walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
                        let mut b = cell.borrow_mut();
                        descend(&mut b, sg, 0, w, d, false)
                    };
                    match self.run_live(segs, &mut wop, &mut walk) {
                        Live::Walked(s, d) => {
                            diags = d;
                            s
                        }
                        Live::Over(r) => return r,
                    }
                } else {
                    let mut b = cell.borrow_mut();
                    let s = descend(&mut b, segs, 0, &mut wop, &mut diags, false);
                    drop(b);
                    s
                }
            }
            RBase::Value(mut v) => {
                if segs.is_empty() {
                    return Ok(match wop {
                        WOp::MakeRef => match v {
                            Value::Reference(r) => Value::Reference(r),
                            other => Value::Reference(PhpRef::new(other)),
                        },
                        WOp::FetchRw | WOp::AssignOp(..) => v.deref(),
                        _ => Value::Null,
                    });
                }
                if live {
                    let mut walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
                        descend(&mut v, sg, 0, w, d, false)
                    };
                    match self.run_live(segs, &mut wop, &mut walk) {
                        Live::Walked(s, d) => {
                            diags = d;
                            s
                        }
                        Live::Over(r) => return r,
                    }
                } else {
                    descend(&mut v, segs, 0, &mut wop, &mut diags, false)
                }
            }
            RBase::Globals => {
                if let Some((RSeg::Dim(k), rest)) = segs.split_first() {
                    let k = self.to_zstr(k)?;
                    let g = self.globals.clone();
                    let slot = g.borrow_mut().find_or_add(k.as_bytes()) as usize;
                    let base = RBase::Var(g, slot);
                    return self.write_place_inner(base, rest, wop, quiet);
                }
                return Err(self.make_throw(
                    "Error",
                    "$GLOBALS can only be modified using the $GLOBALS[$name] = $value syntax",
                ));
            }
            RBase::Undef(_) => return Ok(Value::Null),
        };
        self.finish_write_walk(step, segs, wop, again, diags, quiet)
    }

    /// The end of a [`Vm::write_place_inner`] walk that reached `step`:
    /// the walk's diagnostics (dropped when `quiet`), the write-back of a
    /// compound assignment's result with a second, silent walk from `again`
    /// (the base, kept when such a walk may be needed), and the finished
    /// step.
    fn finish_write_walk(
        &mut self,
        step: Step,
        segs: &[RSeg],
        wop: WOp,
        again: Option<RBase>,
        diags: Diags,
        quiet: bool,
    ) -> VmResult<Value> {
        let appended = if matches!(wop, WOp::AssignOp(..)) {
            take_appended_segs(segs)
        } else {
            None
        };
        let flushed = if quiet {
            Ok(())
        } else {
            self.flush_diags(diags)
        };
        if let Err(t) = flushed {
            return Err(match again {
                Some(b) => self.undo_rw_insert(b, segs, t),
                None => {
                    rw_container_changed(&wop);
                    t
                }
            });
        }
        let changed = rw_container_changed(&wop);
        if changed {
            return Ok(Value::Null);
        }
        match (&step, &wop, again) {
            (Step::Done(cur), WOp::AssignOp(op, rhs), Some(again)) => {
                // The final slot lies inside the base's container: write the
                // result back with a second, silent walk.
                let nv = self.assign_op_value(*op, cur, rhs)?;
                let segs = appended.as_deref().unwrap_or(segs);
                return self.write_place_inner(again, segs, WOp::Assign(nv), true);
            }
            (Step::Err(WalkErr::StringOffsetObject), _, Some(again)) => {
                return self.assign_string_offset_object(&wop, |vm, w| {
                    vm.write_place_inner(again, segs, w, true)
                });
            }
            _ => {}
        }
        self.finish_step(step, segs, wop)
    }

    /// `zend_undefined_offset_write` / `zend_undefined_index_write`: when
    /// the "Undefined array key" warning throws (an error handler), the
    /// missing key is not added. The walk added it before reporting, so a
    /// silent unset walk removes it again, unless the handler already
    /// replaced the array (then only the orphaned copy holds it).
    #[cold]
    #[inline(never)]
    fn undo_rw_insert(&mut self, base: RBase, segs: &[RSeg], err: Throw) -> Throw {
        let unchanged = RW_UNDEF_WATCH
            .with(std::cell::Cell::take)
            .is_some_and(|a| a.refcount() > 1);
        if unchanged {
            let _ = self.write_place_inner(base, segs, WOp::Unset, true);
        }
        err
    }

    /// `zend_fetch_var_address_helper` with `BP_VAR_RW` on an undefined
    /// `$GLOBALS['name']` (`ZEND_FETCH_GLOBAL`): "Undefined global
    /// variable", then the variable is defined as null unless that threw.
    #[cold]
    #[inline(never)]
    fn rw_undefined_global(&mut self, pid: PlaceId, base: &RBase) -> VmResult<()> {
        let RBase::Var(store, idx) = base else {
            return Ok(());
        };
        let name = {
            let st = store.borrow();
            if st.slots.get(*idx).is_none_or(Option::is_some) {
                return Ok(());
            }
            st.names[*idx].clone()
        };
        let msg = self.undefined_place_var_msg(pid, &name);
        self.warning(&msg)?;
        if let Some(slot) = store.borrow_mut().slots.get_mut(*idx) {
            if slot.is_none() {
                *slot = Some(Value::Null);
            }
        }
        Ok(())
    }

    /// A compound assignment to the variable or static property `base`
    /// whose operation threw `err`: a typed reference or a typed static
    /// property takes the checked UNDEF result
    /// (`zend_binary_assign_op_typed_ref()`,
    /// `zend_binary_assign_op_typed_prop()`, Zend/zend_execute.c); any
    /// other variable stays as it was (`zend_binary_op(var_ptr, var_ptr,
    /// value)` leaves `op1`). Returns the exception to throw.
    #[cold]
    #[inline(never)]
    fn failed_assign_op_root(&mut self, base: &RBase, err: Throw) -> Throw {
        match base {
            RBase::Cell(cell) => self.failed_assign_op_static(cell, err),
            RBase::Var(store, idx) => {
                let held = match store.borrow().slots.get(*idx) {
                    Some(Some(Value::Reference(r))) => Some(r.clone()),
                    _ => None,
                };
                match held {
                    Some(r) => self.failed_assign_op_ref(&r, err),
                    None => err,
                }
            }
            _ => err,
        }
    }

    /// [`Vm::failed_assign_op_root`] for compiled variable `cv` of the
    /// frame whose variables are `vars`, when that frame is still the
    /// current one.
    #[cold]
    #[inline(never)]
    fn failed_assign_op_cv(
        &mut self,
        cv: u32,
        vars: *const RefCell<VarStore>,
        err: Throw,
    ) -> Throw {
        let held = self
            .frames
            .last()
            .filter(|f| std::ptr::eq(Rc::as_ptr(&f.vars), vars))
            .and_then(|f| f.cvs.get(cv as usize))
            .and_then(|slot| match slot.get() {
                Some(Value::Reference(r)) => Some(r.clone()),
                _ => None,
            });
        match held {
            Some(r) => self.failed_assign_op_ref(&r, err),
            None => err,
        }
    }

    /// The binary operation of a compound assignment.
    fn assign_op_value(&mut self, op: BinaryOp, cur: &Value, rhs: &Value) -> VmResult<Value> {
        ops::binary_op(op, cur, rhs, self).map_err(|e| self.rt_throw(e))
    }

    /// Finishes a walk step taken inside a container reached through an
    /// object (a property slot, a reference or a handler's result). For
    /// [`WOp::AssignOp`] a final slot's current value comes back as
    /// `Step::Done`: the result is computed with all borrows released and
    /// written by walking the same container again (`walk`) silently.
    fn finish_native(
        &mut self,
        step: Step,
        segs: &[RSeg],
        wop: WOp,
        appended: Option<Vec<RSeg>>,
        mut walk: impl FnMut(&[RSeg], &mut WOp, &mut Diags) -> Step,
    ) -> VmResult<Value> {
        if let (Step::Done(cur), WOp::AssignOp(op, rhs)) = (&step, &wop) {
            let nv = self.assign_op_value(*op, cur, rhs)?;
            let mut w = WOp::Assign(nv);
            let segs = appended.as_deref().unwrap_or(segs);
            let s = walk(segs, &mut w, &mut Vec::new());
            return self.finish_step(s, segs, w);
        }
        if matches!(step, Step::Err(WalkErr::StringOffsetObject)) {
            return self.assign_string_offset_object(&wop, |vm, mut w| {
                let s = walk(segs, &mut w, &mut Vec::new());
                vm.finish_step(s, segs, w)
            });
        }
        self.finish_step(step, segs, wop)
    }

    /// Walks inside one container (`walk` borrows it for the duration of
    /// a call), reports the walk's diagnostics and finishes the step.
    fn descend_with(
        &mut self,
        segs: &[RSeg],
        wop: WOp,
        mut walk: impl FnMut(&[RSeg], &mut WOp, &mut Diags) -> Step,
    ) -> VmResult<Value> {
        let mut w = wop;
        // The write-back walk of `++`/`--` (see `QUIET_AT`).
        let quiet = QUIET_AT.with(std::cell::Cell::get) == Some(self.frames.len());
        let (s, diags) = if !quiet && self.live_walk(segs, &w) {
            match self.run_live(segs, &mut w, &mut walk) {
                Live::Walked(s, d) => (s, d),
                Live::Over(r) => return r,
            }
        } else {
            let mut diags = Vec::new();
            let s = walk(segs, &mut w, &mut diags);
            (s, diags)
        };
        let appended = if matches!(w, WOp::AssignOp(..)) {
            take_appended_segs(segs)
        } else {
            None
        };
        let flushed = if quiet {
            Ok(())
        } else {
            self.flush_diags(diags)
        };
        let changed = rw_container_changed(&w);
        flushed?;
        if changed {
            return Ok(Value::Null);
        }
        self.finish_native(s, segs, w, appended, walk)
    }

    /// Whether a write walk over `segs` doing `wop` runs live: an error
    /// handler could observe its diagnostics (see [`Vm::run_live`]).
    /// Unset walks and walks appending with `[]` (whose repeated walk would
    /// append again) keep collecting them.
    #[inline]
    fn live_walk(&self, segs: &[RSeg], wop: &WOp) -> bool {
        !self.error_handlers.is_empty()
            && !matches!(wop, WOp::Unset)
            && !segs.iter().any(|s| matches!(s, RSeg::Append))
    }

    /// Runs a write walk of one container (`walk`) the way php-src's
    /// opcodes go through it while an error handler is set: the walk stops
    /// at each diagnostic php-src raises in the middle of a fetch
    /// ([`Step::Pause`]), which is raised here with the container held;
    /// then, as php-src decides after it, the walk is repeated past it
    /// (the steps before it are done already and do not repeat their
    /// diagnostics), the fetch is given up (the handler freed or shared the
    /// array: the rest of the place is walked on a temporary null, as
    /// php-src's next opcodes get a null result), or the handler's
    /// exception ends the operation once the fetch it interrupted is
    /// complete.
    #[cold]
    #[inline(never)]
    fn run_live(
        &mut self,
        segs: &[RSeg],
        wop: &mut WOp,
        walk: &mut dyn FnMut(&[RSeg], &mut WOp, &mut Diags) -> Step,
    ) -> Live {
        let (mut seg, mut phase) = (0, 0);
        loop {
            let mut diags = Vec::new();
            let step = with_mode(Mode::Live { seg, phase }, || walk(segs, wop, &mut diags));
            let p = match step {
                Step::Pause(p) => p,
                step => return Live::Walked(step, diags),
            };
            if let Err(t) = self.flush_diags(diags) {
                return Live::Over(Err(t));
            }
            let (at, ph, level, msg, held, check) = match *p {
                Pause::Str(s) => return Live::Over(self.live_string_assign(s, segs, walk)),
                Pause::Fetch {
                    seg,
                    phase,
                    level,
                    msg,
                    held,
                    check,
                } => (seg, phase, level, msg, held, check),
            };
            let owners = |vm: &Self| held.refcount() - vm.symtab_mirrors(&held);
            let before = owners(self);
            let raised = self.error(level, &msg);
            let gone = match check {
                // slow_index_convert_w / zend_undefined_index_write:
                // `GC_DELREF(ht) != 1` — a freed OR a captured/shared array
                // both abandon the write (the handler took a reference).
                // A variable mirrored in both a frame's CV and its attached
                // symbol table counts once (php-src's table holds an
                // `IS_INDIRECT` slot; bug79793).
                Check::Index => owners(self) != before,
                // zend_false_to_array_deprecated: given up only when the
                // handler freed the array (`GC_DELREF(ht) == 0`).
                Check::False | Check::Value => held.refcount() == 1,
            };
            drop(held);
            match raised {
                Err(t) => {
                    if !gone && check != Check::Index {
                        unwind_live(segs, wop, walk, at, ph);
                    }
                    return Live::Over(Err(t));
                }
                Ok(()) if gone => return Live::Over(self.live_give_up(at, check, segs, wop)),
                Ok(()) => (seg, phase) = (at, ph),
            }
        }
    }

    /// The array a `$GLOBALS` read yields (`ZEND_FETCH_GLOBALS` ->
    /// `zend_proptable_to_symtable(&EG(symbol_table), 1)`), as
    /// [`VarStore::to_array`] builds it, except that a reference counts as
    /// held by the table alone (refcount 1 in php-src, so it is copied as
    /// its value) also while compiled variables of frames on the global
    /// scope hold handles of it: such a variable and the table's slot of
    /// the same name are one zval in php-src, reached from the table
    /// through `IS_INDIRECT` ([`Vm::global_ref_mirrors`]).
    fn globals_array(&self) -> Value {
        let st = self.globals.borrow();
        // Sized once (`zend_array_dup` of the symbol table): grown from
        // empty, the copy of a few thousand globals rehashed a dozen times
        // (`array_key_exists($name, $GLOBALS)` of MediaWiki's config).
        let mut t = php_runtime::HashTable::with_capacity_mixed(st.names.len());
        for (n, v) in st.names.iter().zip(&st.slots) {
            let Some(v) = v else { continue };
            let v = match v {
                Value::Reference(r)
                    if r.refcount() == 1 || r.refcount() == 1 + self.global_ref_mirrors(n, r) =>
                {
                    r.get()
                }
                other => other.clone(),
            };
            // The names of a symbol table are unique: a name that is no
            // integer in digits is appended without the probe.
            if ArrayKey::surely_str(n.as_bytes()) {
                t.add_new_str(n.clone(), v);
            } else {
                t.insert(ArrayKey::from_zstr(n), v);
            }
        }
        Value::Array(ZArray::from_table(t))
    }

    /// The handles of reference `r`, the global symbol table's value of
    /// variable `name`, that are the same variable's: the compiled variable
    /// `name` of each frame running on the global scope (attached CVs keep
    /// a second handle of their table slot's value, see
    /// [`Frame::attach_symbol_table`](crate::vm::Frame::attach_symbol_table)).
    #[cold]
    fn global_ref_mirrors(&self, name: &ZStr, r: &PhpRef) -> usize {
        self.frames
            .iter()
            .filter(|f| Rc::ptr_eq(&f.vars, &self.globals))
            .filter(|f| {
                f.code
                    .cv_names
                    .iter()
                    .position(|n| n.as_bytes() == name.as_bytes())
                    .and_then(|i| f.cvs.get(i))
                    .and_then(crate::vm::CvSlot::get)
                    .is_some_and(|v| matches!(v, Value::Reference(c) if PhpRef::ptr_eq(c, r)))
            })
            .count()
    }

    /// The duplicate handles of `arr` that the VM's two copies of a
    /// variable make: a compiled variable of a frame with an attached
    /// symbol table and that table's slot of the same name both holding
    /// `arr` are one owner in php-src, whose table slot is `IS_INDIRECT`.
    #[cold]
    fn symtab_mirrors(&self, arr: &ZArray) -> usize {
        let holds =
            |v: Option<&Value>| matches!(v, Some(Value::Array(a)) if ZArray::ptr_eq(a, arr));
        let mut n = 0;
        for f in self.frames.iter().filter(|f| f.symtab_attached) {
            let Ok(mut st) = f.vars.try_borrow_mut() else {
                continue;
            };
            for (i, name) in f.code.cv_names.iter().enumerate() {
                if !holds(f.cvs.get(i).and_then(crate::vm::CvSlot::get)) {
                    continue;
                }
                let slot = st.find(name.as_bytes()).map(|s| s as usize);
                if slot.is_some_and(|s| holds(st.slots.get(s).and_then(Option::as_ref))) {
                    n += 1;
                }
            }
        }
        n
    }

    /// The fetch of segment `seg` was given up after a diagnostic (the
    /// error handler freed or shared the array, see [`Vm::run_live`]): the
    /// opcode yields null, the following ones work on that temporary, and
    /// a final assignment does not happen.
    #[cold]
    #[inline(never)]
    fn live_give_up(
        &mut self,
        seg: usize,
        check: Check,
        segs: &[RSeg],
        wop: &mut WOp,
    ) -> VmResult<Value> {
        let last = seg + 1 == segs.len();
        if check == Check::False
            && (!last || matches!(wop, WOp::FetchRw | WOp::MakeRef | WOp::AssignRef(_)))
        {
            // `zend_fetch_dimension_address`'s `return_null` still reads an
            // undefined-variable offset.
            if let Some(name) = take_pending(segs, seg) {
                self.warning(&crate::exec::undefined_var_msg(&name))?;
            }
        }
        // Read by `Vm::fetch_rw` for a place ending in a dimension (a
        // property's `++` on the temporary throws instead).
        let rw_fetch =
            matches!(wop, WOp::FetchRw) && matches!(segs.last(), Some(RSeg::Dim(_) | RSeg::Append));
        let r = if last {
            match wop {
                WOp::MakeRef => Ok(Value::Reference(PhpRef::new(Value::Null))),
                _ => Ok(Value::Null),
            }
        } else {
            let w = std::mem::replace(wop, WOp::Unset);
            self.descend_temp(Value::Null, segs, seg + 1, w)
        };
        if rw_fetch {
            // `++`/`--` then act on the temporary (see `Vm::fetch_rw`).
            RW_SKIPPED.with(|c| c.set(true));
        }
        r
    }

    /// php-src's `zend_assign_to_string_offset` for a final assignment to a
    /// string offset that raises a diagnostic, raised here with the
    /// (separated) string held: when the error handler frees it the write
    /// is given up and the assignment yields null. The byte is then written
    /// by walking again (`walk`) with the one-byte value.
    #[cold]
    #[inline(never)]
    fn live_string_assign(
        &mut self,
        p: StrAssign,
        segs: &[RSeg],
        walk: &mut dyn FnMut(&[RSeg], &mut WOp, &mut Diags) -> Step,
    ) -> VmResult<Value> {
        let StrAssign {
            held,
            dim,
            undef_dim,
            value,
            undef_value,
        } = p;
        let offset = if let Value::Int(i) = dim {
            i
        } else {
            // `zend_check_string_offset`.
            let mut d = Vec::new();
            if let Some(name) = &undef_dim {
                d.push((E_WARNING, crate::exec::undefined_var_msg(name)));
            }
            let off = string_offset(&dim, &mut d, false);
            self.flush_diags(d)?;
            if held.refcount() == 1 {
                return Ok(Value::Null);
            }
            match off {
                Ok(o) => o,
                Err(e) => return Err(self.walk_err(e)),
            }
        };
        let len = i64::try_from(held.len()).unwrap_or(i64::MAX);
        if offset < -len {
            self.warning(&format!("Illegal string offset {offset}"))?;
            return Ok(Value::Null);
        }
        let bytes = match value.deref() {
            Value::Str(s) => s.as_bytes().to_vec(),
            other => {
                // `zval_try_get_string_func`, with the string held across
                // the value's warnings.
                if let Some(name) = &undef_value {
                    self.warning(&crate::exec::undefined_var_msg(name))?;
                }
                let bytes = match &other {
                    Value::Array(_) => {
                        self.warning("Array to string conversion")?;
                        b"Array".to_vec()
                    }
                    Value::Object(_) => self.to_zstr(&other)?.as_bytes().to_vec(),
                    v => string_offset_value_bytes(v),
                };
                if held.refcount() == 1 {
                    return Ok(Value::Null);
                }
                bytes
            }
        };
        let Some(&c) = bytes.first() else {
            return Err(
                self.make_throw("Error", "Cannot assign an empty string to a string offset")
            );
        };
        if bytes.len() > 1 {
            self.warning("Only the first byte will be assigned to the string offset")?;
            if held.refcount() == 1 {
                return Ok(Value::Null);
            }
        }
        drop(held);
        let mut w = WOp::Assign(Value::Str(ZStr::from_byte(c)));
        let s = walk(segs, &mut w, &mut Vec::new());
        self.finish_step(s, segs, w)
    }

    /// Continues a write walk at segment `i` inside a reference.
    fn descend_ref(&mut self, r: &PhpRef, segs: &[RSeg], i: usize, wop: WOp) -> VmResult<Value> {
        self.descend_with(segs, wop, |sg, w, d| {
            let mut b = r.borrow_mut();
            descend(&mut b, sg, i, w, d, false)
        })
    }

    /// Continues a write walk at segment `i` inside a temporary (a
    /// handler's non-reference result: the modification is lost).
    fn descend_temp(&mut self, mut v: Value, segs: &[RSeg], i: usize, wop: WOp) -> VmResult<Value> {
        self.descend_with(segs, wop, |sg, w, d| descend(&mut v, sg, i, w, d, false))
    }

    fn finish_step(&mut self, step: Step, segs: &[RSeg], wop: WOp) -> VmResult<Value> {
        match step {
            Step::Done(v) => Ok(v),
            Step::Err(e) => Err(self.walk_err(e)),
            Step::TypedRefAssign(r, v) => self.assign_ref(&r, v),
            Step::Obj(o, i) => self.obj_walk(o, segs, i, wop),
            // Only live walks pause, and `Vm::run_live` handles that.
            Step::Pause(p) => match *p {
                Pause::Fetch { level, msg, .. } => {
                    self.error(level, &msg)?;
                    Ok(Value::Null)
                }
                Pause::Str(_) => Ok(Value::Null),
            },
        }
    }

    /// [`WOp::AssignOp`] on variable slot `idx` of `store`: the current
    /// value (an undefined variable warns and is null, see
    /// [`Vm::rw_undefined_var`]), the operation, and the result assigned.
    fn write_var_root_assign_op(
        &mut self,
        store: &Rc<RefCell<VarStore>>,
        idx: usize,
        op: BinaryOp,
        rhs: &Value,
    ) -> VmResult<Value> {
        let mut held = None;
        let cur = store.borrow().slots[idx].as_ref().map(Value::deref);
        let cur = if let Some(v) = cur {
            v
        } else {
            let name = store.borrow().names[idx].clone();
            held = self.rw_undefined_var(&name)?;
            Value::Null
        };
        let nv = match self.assign_op_value(op, &cur, rhs) {
            Ok(nv) => nv,
            Err(e) => {
                let in_place = concats_in_place(op, &cur);
                drop(cur);
                return Err(match held {
                    Some(t) => t,
                    None if in_place => e,
                    None => self.failed_assign_op_root(&RBase::Var(store.clone(), idx), e),
                });
            }
        };
        let r = self.write_var_root(store, idx, WOp::Assign(nv))?;
        match held {
            Some(t) => Err(t),
            None => Ok(r),
        }
    }

    /// The "Undefined variable" warning of a read-modify-write of the
    /// undefined variable `name` (`$x op= v`, `$x++`). php-src has set the
    /// variable to null first (`_get_zval_cv_lookup_BP_VAR_RW`) and the
    /// opcode goes on with an exception an error handler threw: the result
    /// is stored, then the exception is thrown. Returns that exception; an
    /// exit or a fatal error stops at once.
    #[cold]
    fn rw_undefined_var(&mut self, name: &ZStr) -> VmResult<Option<Throw>> {
        match self.warning(&crate::exec::undefined_var_msg(name)) {
            Ok(()) => Ok(None),
            Err(t @ Throw::Exception(_)) => Ok(Some(t)),
            Err(t) => Err(t),
        }
    }

    fn write_var_root(
        &mut self,
        store: &Rc<RefCell<VarStore>>,
        idx: usize,
        wop: WOp,
    ) -> VmResult<Value> {
        if !matches!(wop, WOp::FetchRw | WOp::AssignOp(..)) {
            self.clear_cvs_for_slot(store, idx);
        }
        match wop {
            WOp::Assign(v) => {
                let mut st = store.borrow_mut();
                if let Some(Value::Reference(r)) = &st.slots[idx] {
                    if crate::typed_refs::has_type_sources(r) {
                        let r = r.clone();
                        drop(st);
                        return self.assign_ref(&r, v);
                    }
                }
                let old = match &mut st.slots[idx] {
                    Some(Value::Reference(r)) => Some(r.set(v.clone())),
                    slot => slot.replace(v.clone()),
                };
                drop(st);
                if let Some(old) = &old {
                    crate::gc::possible_root(old);
                }
                drop(old);
                Ok(v)
            }
            WOp::FetchRw => {
                let cur = store.borrow().slots[idx].as_ref().map(Value::deref);
                if let Some(v) = cur {
                    return Ok(v);
                }
                let name = store.borrow().names[idx].clone();
                if let Some(t) = self.rw_undefined_var(&name)? {
                    RW_THROW.with(|c| c.set(Some(t)));
                }
                Ok(Value::Null)
            }
            // `$a += $b` of two arrays (`add_function_array` with result ==
            // op1): the union is made in the variable's own array, which is
            // separated first only if something else shares it, instead of
            // in a copy. Nothing is released and no user code runs.
            WOp::AssignOp(BinaryOp::Add, Value::Array(rhs)) => {
                let mut st = store.borrow_mut();
                if let Some(Some(Value::Array(a))) = st.slots.get_mut(idx) {
                    if !ZArray::ptr_eq(a, &rhs) {
                        a.make_mut().merge(&rhs, false);
                    }
                    return Ok(Value::Array(a.clone()));
                }
                drop(st);
                self.write_var_root_assign_op(store, idx, BinaryOp::Add, &Value::Array(rhs))
            }
            WOp::AssignOp(op, rhs) => self.write_var_root_assign_op(store, idx, op, &rhs),
            WOp::MakeRef => {
                let mut st = store.borrow_mut();
                let slot = st.slots[idx].get_or_insert(Value::Null);
                Ok(Value::Reference(slot.make_ref()))
            }
            WOp::AssignRef(r) => {
                let old = store.borrow_mut().slots[idx].replace(r.clone());
                if let Some(old) = &old {
                    crate::gc::possible_root(old);
                }
                drop(old);
                Ok(r)
            }
            WOp::Unset => {
                let old = store.borrow_mut().slots[idx].take();
                if let Some(old) = &old {
                    crate::gc::possible_root(old);
                }
                drop(old);
                Ok(Value::Null)
            }
            w @ WOp::AssignDeref(_) => self.write_var_root(store, idx, w.resolve_deferred()),
        }
    }

    /// `ZEND_ASSIGN_OBJ_OP` with `.=` on an initialized plain public
    /// property slot holding a string ([`Vm::plain_property_slot`]: the
    /// slot `get_property_ptr_ptr` hands out, with no hook, magic method,
    /// native handler, lazy state or type involved), for the places the
    /// property cache does not cover (`$this->o->s .= 'x'`,
    /// `$list[$i]->name .= 'x'`): the string is extended in place like
    /// [`concat_in_place`] does. `None` leaves the operation to the generic
    /// read, operate and write-back path.
    fn plain_prop_concat(o: &Obj, name: &ZStr, rhs: &Value) -> Option<Value> {
        if !matches!(rhs, Value::Str(_) | Value::Int(_)) {
            return None;
        }
        let idx = Self::plain_property_slot(o, name, true)?;
        let data = object::data(o);
        let mut slots = data.slots.borrow_mut();
        let r = match (slots.get_mut(idx), rhs) {
            (Some(Some(Value::Str(s))), Value::Str(add)) => {
                s.append(add.as_bytes());
                Some(Value::Str(s.clone()))
            }
            // The digits go straight into the string (`ZStr::append_long`).
            (Some(Some(Value::Str(s))), Value::Int(i)) => {
                s.append_long(*i);
                Some(Value::Str(s.clone()))
            }
            _ => None,
        };
        drop(slots);
        r
    }

    /// Continues a write walk at an object.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn obj_walk(
        &mut self,
        obj: Obj,
        segs: &[RSeg],
        i: usize,
        wop: WOp,
    ) -> VmResult<Value> {
        let last = i + 1 == segs.len();
        match &segs[i] {
            RSeg::Prop(name) => {
                if last {
                    return match wop {
                        WOp::Assign(v) => {
                            // ZEND_ASSIGN_OBJ reads OP_DATA once it has the
                            // object.
                            if v.is_null() {
                                self.take_pending_value_warning(segs)?;
                            }
                            self.write_property_walk(&obj, name, v)
                        }
                        WOp::FetchRw => match self.native_read_property(
                            &obj,
                            name,
                            crate::ext::PropCtx::ReadWrite,
                        )? {
                            Some(v) => Ok(v.deref()),
                            None => self.fetch_property_rw(&obj, name, RwThrow::IncDec),
                        },
                        WOp::AssignOp(op, rhs) => {
                            if op == BinaryOp::Concat {
                                if let Some(r) = Self::plain_prop_concat(&obj, name, &rhs) {
                                    return Ok(r);
                                }
                            }
                            // `zend_assign_op_overloaded_property` and the
                            // `get_property_ptr_ptr` path alike: read, operate,
                            // write back through the property handlers.
                            let (cur, native) = match self.native_read_property(
                                &obj,
                                name,
                                crate::ext::PropCtx::ReadWrite,
                            )? {
                                Some(v) => (v.deref(), true),
                                None => (
                                    self.fetch_property_rw(&obj, name, RwThrow::Op(op, &rhs))?,
                                    false,
                                ),
                            };
                            let nv = match self.assign_op_value(op, &cur, &rhs) {
                                Ok(nv) => nv,
                                // `zend_binary_assign_op_typed_prop` keeps
                                // the checked UNDEF result; the handlers'
                                // path writes only on success.
                                Err(t) if native || concats_in_place(op, &cur) => return Err(t),
                                Err(t) => {
                                    drop(cur);
                                    return Err(self.failed_assign_op_prop(&obj, name, t));
                                }
                            };
                            drop(cur);
                            self.write_property(&obj, name, nv)
                        }
                        WOp::MakeRef => self.property_ref(&obj, name).map(Value::Reference),
                        WOp::AssignRef(r) => self.assign_property_ref(&obj, name, r),
                        WOp::Unset => {
                            self.unset_property(&obj, name)?;
                            Ok(Value::Null)
                        }
                        w @ WOp::AssignDeref(_) => {
                            self.obj_walk(obj, segs, i, w.resolve_deferred())
                        }
                    };
                }
                // Intermediate: native classes provide the container through
                // read_property (`get_property_ptr_ptr` returning NULL).
                let ctx = match wop {
                    WOp::Unset => crate::ext::PropCtx::Unset,
                    WOp::FetchRw | WOp::AssignOp(..) => crate::ext::PropCtx::ReadWrite,
                    _ => crate::ext::PropCtx::Write,
                };
                // Objects and references continue the walk; other values
                // fall back to the standard property storage.
                match self.native_read_property(&obj, name, ctx)? {
                    Some(Value::Object(o2)) => return self.obj_walk(o2, segs, i + 1, wop),
                    Some(Value::Reference(r)) => return self.descend_ref(&r, segs, i + 1, wop),
                    // A native property without usable storage (virtual or
                    // write-restricted): as in `zend_fetch_property_address`
                    // with `get_property_ptr_ptr` returning NULL, the walk goes
                    // on with `read_property`'s temporary (DOM's
                    // `$node->previousSibling->nodeValue = ''` on a null
                    // sibling: "Attempt to assign property on null").
                    Some(other)
                        if !matches!(wop, WOp::Unset)
                            && Self::native_prop_without_ptr(&obj, name) =>
                    {
                        return self.descend_temp(other, segs, i + 1, wop);
                    }
                    _ => {}
                }
                if matches!(wop, WOp::Unset) {
                    let v = self.read_property_for_unset(&obj, name)?;
                    return match v {
                        Some(Value::Object(o2)) => self.obj_walk(o2, segs, i + 1, wop),
                        Some(Value::Reference(r)) => {
                            let mut diags = Vec::new();
                            let mut w = wop;
                            let s = {
                                let mut b = r.borrow_mut();
                                descend(&mut b, segs, i + 1, &mut w, &mut diags, false)
                            };
                            self.flush_diags(diags)?;
                            self.finish_step(s, segs, w)
                        }
                        // `ZEND_FETCH_OBJ_UNSET` hands `ZEND_UNSET_DIM` &
                        // co. the property's value: a string throws
                        // "Cannot unset string offsets", another scalar
                        // "Cannot unset offset in a non-array variable",
                        // `false` deprecates, null unsets nothing.
                        Some(other) => self.descend_temp(other, segs, i + 1, wop),
                        None => Ok(Value::Null),
                    };
                }
                self.typed_prop_dim_write_check(&obj, name, &segs[i + 1], &wop)?;
                let rw = matches!(wop, WOp::FetchRw | WOp::AssignOp(..));
                match self.property_container(&obj, name, rw, 1)? {
                    PropContainer::Declared(idx) => {
                        let obj = self.lazy_object_for(&obj, Some(name))?;
                        let walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
                            let data = object::data(&obj);
                            let mut slots = data.slots.borrow_mut();
                            let slot = slots[idx].get_or_insert(Value::Null);
                            descend(slot, sg, i + 1, w, d, false)
                        };
                        self.descend_with(segs, wop, walk)
                    }
                    PropContainer::Dynamic(key) => {
                        let obj = self.lazy_object_for(&obj, Some(name))?;
                        let walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
                            let data = object::data(&obj);
                            let mut dynp = data.dynamic.borrow_mut();
                            let t = dynp.get_or_insert_with(Box::default);
                            let slot = t.lookup(key.clone());
                            descend(slot, sg, i + 1, w, d, false)
                        };
                        self.descend_with(segs, wop, walk)
                    }
                    PropContainer::Value(v) => match v {
                        Value::Object(o2) => self.obj_walk(o2, segs, i + 1, wop),
                        other => self.descend_temp(other, segs, i + 1, wop),
                    },
                }
            }
            seg @ (RSeg::Dim(_) | RSeg::Append) => {
                // An undefined-variable offset is read before the object's
                // handlers run (`zend_fetch_dimension_address`,
                // `ZEND_ASSIGN_DIM`, `zend_binary_assign_op_obj_dim`).
                // An exception the error handler throws there is held: the
                // opcode goes on, and the "Cannot use object ... as array"
                // error of an object without ArrayAccess is chained onto it
                // (`zend_throw_error` while `EG(exception)` is set).
                let mut held = None;
                if matches!(seg, RSeg::Dim(Value::Null)) {
                    if let Some(name) = take_pending(segs, i) {
                        held = self.rw_undefined_var(&name)?;
                    }
                }
                // `zend_binary_assign_op_obj_dim` reads OP_DATA next (a
                // warning raised while an exception is pending reaches no
                // handler: `zend_call_function` returns at once).
                if last && matches!(wop, WOp::AssignOp(..)) && held.is_none() {
                    held = self.held_value_warning(segs)?;
                }
                if let Some(t) = held {
                    if crate::ext::hooks_of(&obj).is_some() {
                        let keyopt = match seg {
                            RSeg::Dim(k) => Some(k),
                            _ => None,
                        };
                        if !last {
                            return Err(t);
                        }
                        return Err(self.native_dim_write_held(&obj, keyopt, &wop, t));
                    }
                    if self.instance_of(&obj, b"arrayaccess") {
                        return Err(t);
                    }
                    let assign_op = last && matches!(wop, WOp::AssignOp(..));
                    return Err(self.object_as_array_error(&obj, Some(t), assign_op));
                }
                let key = match seg {
                    RSeg::Dim(k) => k.clone(),
                    _ => Value::Null,
                };
                if let Some(h) = crate::ext::hooks_of(&obj) {
                    let keyopt = match seg {
                        RSeg::Dim(k) => Some(k),
                        _ => None,
                    };
                    if last {
                        match &wop {
                            WOp::Assign(v) => {
                                if let Some(f) = h.write_dimension {
                                    if v.is_null() {
                                        // `ZEND_ASSIGN_DIM` goes on with an
                                        // exception the "Undefined
                                        // variable" warning threw.
                                        if let Some(t) = self.held_value_warning(segs)? {
                                            return Err(
                                                self.native_dim_write_held(&obj, keyopt, &wop, t)
                                            );
                                        }
                                    }
                                    if f(self, &obj, keyopt, v)? {
                                        return Ok(v.clone());
                                    }
                                }
                            }
                            WOp::AssignRef(r) => {
                                if keyopt.is_some() {
                                    if let Some(f) = h.write_dimension {
                                        if f(self, &obj, keyopt, r)? {
                                            return Ok(r.clone());
                                        }
                                    }
                                }
                            }
                            WOp::Unset => {
                                if let (Some(f), Some(k)) = (h.unset_dimension, keyopt) {
                                    if f(self, &obj, k)? {
                                        return Ok(Value::Null);
                                    }
                                }
                            }
                            WOp::AssignOp(op, rhs) => {
                                // `zend_binary_assign_op_obj_dim`: read_dimension
                                // with BP_VAR_R (the hook context outside hook
                                // calls), operate, then write_dimension.
                                let op_read_key =
                                    if keyopt.is_none() { Some(&key) } else { keyopt };
                                if let (Some(f), Some(k)) = (h.read_dimension, op_read_key) {
                                    if let Some(cur) = f(self, &obj, k)? {
                                        let nv = self.assign_op_value(*op, &cur.deref(), rhs)?;
                                        drop(cur);
                                        return self.obj_walk(
                                            obj.clone(),
                                            segs,
                                            i,
                                            WOp::Assign(nv),
                                        );
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    // `$o[][..] = v` and `&$o[]` read the appended dimension
                    // (no offset).
                    let read_key = if keyopt.is_none()
                        && (!last
                            || matches!(
                                wop,
                                WOp::MakeRef | WOp::AssignRef(_) | WOp::FetchRw | WOp::AssignOp(..)
                            )) {
                        Some(&key)
                    } else {
                        keyopt
                    };
                    if let (Some(f), Some(k)) = (h.read_dimension, read_key) {
                        if !last
                            || matches!(
                                wop,
                                WOp::FetchRw | WOp::MakeRef | WOp::AssignRef(_) | WOp::AssignOp(..)
                            )
                        {
                            let rw = matches!(wop, WOp::FetchRw | WOp::AssignOp(..));
                            self.dim_ctx = match (keyopt.is_none(), rw, &wop, last) {
                                (true, true, ..) => crate::ext::DimensionCtx::AppendRw,
                                (true, false, ..) => crate::ext::DimensionCtx::Append,
                                (false, true, ..) => crate::ext::DimensionCtx::ReadWrite,
                                (false, false, WOp::Unset, false) => {
                                    crate::ext::DimensionCtx::Unset
                                }
                                _ => crate::ext::DimensionCtx::Write,
                            };
                            let r = f(self, &obj, k);
                            self.dim_ctx = crate::ext::DimensionCtx::Read;
                            if let Some(v) = r? {
                                if last {
                                    return Ok(match wop {
                                        WOp::MakeRef => Value::Reference(match v {
                                            Value::Reference(r) => r,
                                            o => PhpRef::new(o),
                                        }),
                                        WOp::AssignRef(ref r) => {
                                            if let Value::Reference(target_ref) = &v {
                                                self.assign_ref(target_ref, r.clone())?;
                                            } else {
                                                return Err(self.make_throw(
                                                    "Error",
                                                    "Cannot assign by reference to an array dimension of an object",
                                                ));
                                            }
                                            r.clone()
                                        }
                                        _ => v.deref(),
                                    });
                                }
                                return match v {
                                    Value::Object(o2) => self.obj_walk(o2, segs, i + 1, wop),
                                    Value::Reference(rf) => self.descend_ref(&rf, segs, i + 1, wop),
                                    other => self.descend_temp(other, segs, i + 1, wop),
                                };
                            }
                        }
                    }
                }
                if !self.instance_of(&obj, b"arrayaccess") {
                    // `ZEND_ASSIGN_DIM` reads OP_DATA before
                    // `zend_std_write_dimension` throws.
                    let held = if last && matches!(wop, WOp::Assign(Value::Null)) {
                        self.held_value_warning(segs)?
                    } else {
                        None
                    };
                    let assign_op = last && matches!(wop, WOp::AssignOp(..));
                    return Err(self.object_as_array_error(&obj, held, assign_op));
                }
                if last {
                    let assign_ref = matches!(wop, WOp::AssignRef(_));
                    return match wop {
                        WOp::Assign(v) => {
                            if v.is_null() {
                                self.take_pending_value_warning(segs)?;
                            }
                            self.call_method_name_n(&obj, "offsetSet", [key, v.clone()])?;
                            Ok(v)
                        }
                        // `zend_binary_assign_op_obj_dim`: `$o[] op= v` reads
                        // `offsetGet(null)` (bug69955); the write-back then
                        // calls `offsetSet(null, ...)`.
                        WOp::FetchRw => {
                            let v =
                                self.call_method_name_raw_n(&obj, "offsetGet", [key.clone()])?;
                            OVERLOADED_RW.with(|c| c.set(Some((obj, key))));
                            Ok(v)
                        }
                        WOp::AssignOp(op, rhs) => {
                            // `zend_binary_assign_op_obj_dim` over
                            // `zend_std_read_dimension`/`write_dimension`
                            // (`$o[] op= v` passes a null offset to both).
                            let cur =
                                self.call_method_name_raw_n(&obj, "offsetGet", [key.clone()])?;
                            let nv = self.assign_op_value(op, &cur.deref(), &rhs)?;
                            drop(cur);
                            self.call_method_name_n(&obj, "offsetSet", [key, nv.clone()])?;
                            Ok(nv)
                        }
                        WOp::Unset => {
                            self.call_method_name_n(&obj, "offsetUnset", [key])?;
                            Ok(Value::Null)
                        }
                        WOp::MakeRef | WOp::AssignRef(_) => {
                            let r = self.call_method_name_raw_n(&obj, "offsetGet", [key])?;
                            let r = match r {
                                Value::Reference(r) => r,
                                other => {
                                    // `zend_fetch_dimension_address` (W): an
                                    // object result is still usable.
                                    if !matches!(other, Value::Object(_)) {
                                        self.notice(&format!(
                                            "Indirect modification of overloaded element of {} has no effect",
                                            object::class_of(&obj).name
                                        ))?;
                                    }
                                    PhpRef::new(other)
                                }
                            };
                            if assign_ref {
                                // `ZEND_ASSIGN_REF`: the fetched dimension
                                // is not an INDIRECT slot.
                                return Err(self.make_throw(
                                    "Error",
                                    "Cannot assign by reference to an array dimension of an object",
                                ));
                            }
                            Ok(Value::Reference(r))
                        }
                        w @ WOp::AssignDeref(_) => {
                            self.obj_walk(obj, segs, i, w.resolve_deferred())
                        }
                    };
                }
                let r = self.call_method_name_raw_n(&obj, "offsetGet", [key])?;
                match r {
                    Value::Object(o2) => self.obj_walk(o2, segs, i + 1, wop),
                    Value::Reference(rf) => {
                        let inner = rf.get();
                        if let Value::Object(o2) = inner {
                            return self.obj_walk(o2, segs, i + 1, wop);
                        }
                        self.descend_ref(&rf, segs, i + 1, wop)
                    }
                    other => {
                        // An increment walks twice (read, then write): the
                        // write walk reports.
                        if !matches!(wop, WOp::FetchRw) {
                            self.notice(&format!(
                                "Indirect modification of overloaded element of {} has no effect",
                                object::class_of(&obj).name
                            ))?;
                        }
                        self.descend_temp(other, segs, i + 1, wop)
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Public place operations
    // ------------------------------------------------------------------

    /// `place = $cv` where `$cv` is a reference to null, false or an array
    /// ([`crate::exec::CvOpData::Deferred`]). `ZEND_ASSIGN_DIM`
    /// (`Zend/zend_vm_def.h`) fetches the container for writing (creating
    /// and separating arrays) before it reads `OP_DATA`, so when the
    /// reference is a container on the path, the array ends up holding
    /// itself rather than a copy: `$t = &$a; $a[] = $t;` (gh16318,
    /// mb_check_encoding_array). The value is therefore read at the final
    /// slot ([`WOp::AssignDeref`]). Places on a static property or through
    /// a property, and the special offset cases of [`Vm::place_assign`],
    /// read it first, as before.
    #[cold]
    #[inline(never)]
    fn place_assign_cv_ref_impl(&mut self, pid: PlaceId, cv: u32) -> VmResult<Value> {
        // `$a[] = $a` (`zend_is_assign_to_self`): php-src copies the value
        // into a temporary before fetching the container.
        let to_self = matches!(
            self.frame().code.places[pid as usize].base,
            PlaceBase::Cv(b) if b == cv
        );
        let src = match self.cv_get(cv) {
            Some(Value::Reference(r)) if !to_self => r,
            Some(v) => {
                let d = v.deref();
                crate::gc::release_duplicate(v);
                return self.place_assign(pid, d);
            }
            None => return self.place_assign_undef_cv(pid, cv),
        };
        let resolved = self.resolve_place_ex(pid, true, false, PlaceOp::Assign);
        let (base, segs) = match resolved {
            Ok(r) => r,
            Err(e) => {
                crate::gc::release_duplicate(Value::Reference(src));
                return Err(e);
            }
        };
        let target = match &base {
            RBase::Var(store, idx)
                if !segs.is_empty()
                    && !segs.iter().any(|s| matches!(s, RSeg::Prop(_)))
                    && store.borrow().slots.get(*idx).is_some_and(Option::is_some) =>
            {
                Some((store.clone(), *idx))
            }
            _ => None,
        };
        let Some((store, idx)) = target else {
            let v = src.get();
            crate::gc::release_duplicate(Value::Reference(src));
            return self.write_place(base, &segs, WOp::Assign(v));
        };
        self.clear_cvs_for_slot(&store, idx);
        let mut wop = WOp::AssignDeref(Box::new(DeferredRead {
            src,
            hit: DeferredHit::None,
        }));
        let mut diags = Vec::new();
        let walk_slot = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
            let mut st = store.borrow_mut();
            let step = match st.slots.get_mut(idx).and_then(Option::as_mut) {
                Some(slot) => descend(slot, sg, 0, w, d, false),
                None => Step::Done(Value::Null),
            };
            drop(st);
            step
        };
        let step = if self.live_walk(&segs, &wop) {
            let mut walk = walk_slot;
            match self.run_live(&segs, &mut wop, &mut walk) {
                Live::Walked(s, d) => {
                    diags = d;
                    s
                }
                Live::Over(r) => {
                    drop(wop.resolve_deferred());
                    flush_deferred_ref_assigns();
                    self.frame_mut().sync_from_symbol_table();
                    if self.store_in_globals(&store) {
                        self.sync_global_frames();
                    }
                    return r;
                }
            }
        } else {
            let walk = walk_slot;
            walk(&*segs, &mut wop, &mut diags)
        };
        let step = match (step, &mut wop) {
            (Step::Done(v), WOp::AssignDeref(d)) => finish_deferred(v, d, &store, idx, &segs),
            (s, _) => s,
        };
        let flushed = self.flush_diags(diags);
        let wop = wop.resolve_deferred();
        let r = match flushed {
            Ok(()) => self.finish_step(step, &segs, wop),
            Err(e) => Err(e),
        };
        flush_deferred_ref_assigns();
        self.frame_mut().sync_from_symbol_table();
        if self.store_in_globals(&store) {
            self.sync_global_frames();
        }
        r
    }

    /// `place = v`
    fn place_assign_impl(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        let (single, head) = {
            let p = &self.frame().code.places[pid as usize];
            (propcache::cacheable(p), propcache::cacheable_head(p))
        };
        // Fast paths: ASSIGN_OBJ to a cached property, ASSIGN_DIM to an
        // element of a cached array property (`$this->a[$k] = v`).
        let v = if single {
            match self.prop_cache_assign(pid, v) {
                Ok(r) => return Ok(r),
                Err(v) => v,
            }
        } else if head {
            match self.prop_cache_nested_assign(pid, v) {
                Ok(r) => return Ok(r),
                Err(v) => v,
            }
        } else {
            v
        };
        let (base, segs) = self.resolve_place_ex(pid, true, false, PlaceOp::Assign)?;
        // Fast path: `$this->a[$k] = v` & co. enter a cached declared slot
        // (FETCH_OBJ_W with a runtime cache slot).
        if head && !single {
            if let Some((obj, idx)) = self.prop_cache_container(pid) {
                drop(base);
                return self.write_cached_container(&obj, idx, &segs, WOp::Assign(v));
            }
        }
        self.write_place(base, &segs, WOp::Assign(v))
    }

    /// The cached declared slot a write walk of place `pid` (resolved to
    /// `segs`) may start from: `Some` when its first segment is a cached
    /// property and more segments follow (see
    /// [`Vm::write_cached_container`]).
    fn cached_container_of(&mut self, pid: PlaceId, segs: &[RSeg]) -> Option<(Obj, usize)> {
        if segs.len() > 1 && propcache::cacheable_head(&self.frame().code.places[pid as usize]) {
            self.prop_cache_container(pid)
        } else {
            None
        }
    }

    /// [`Vm::write_place`] of place `pid`, starting from a cached first
    /// property when the cache covers it.
    fn write_place_cached(
        &mut self,
        pid: PlaceId,
        base: RBase,
        segs: &[RSeg],
        wop: WOp,
    ) -> VmResult<Value> {
        if let Some((obj, idx)) = self.cached_container_of(pid, segs) {
            drop(base);
            return self.write_cached_container(&obj, idx, segs, wop);
        }
        self.write_place(base, segs, wop)
    }

    /// The write walk of a resolved place whose first segment is a cached
    /// declared property, from slot `idx` of `obj` on: what
    /// [`Vm::write_place`] does for an object base reaching `obj_walk`'s
    /// `PropContainer::Declared(idx)`, once `Vm::prop_cache_container`
    /// stood in for the property's `FETCH_OBJ_W`.
    fn write_cached_container(
        &mut self,
        obj: &Obj,
        idx: usize,
        segs: &[RSeg],
        wop: WOp,
    ) -> VmResult<Value> {
        // `write_place_inner`'s check after the base step.
        if rw_container_changed(&wop) {
            flush_deferred_ref_assigns();
            return Ok(Value::Null);
        }
        let walk = |sg: &[RSeg], w: &mut WOp, d: &mut Diags| {
            let data = object::data(obj);
            let mut slots = data.slots.borrow_mut();
            let slot = slots[idx].get_or_insert(Value::Null);
            descend(slot, sg, 1, w, d, false)
        };
        let r = self.descend_with(segs, wop, walk);
        flush_deferred_ref_assigns();
        r
    }

    /// The `BP_VAR_RW` read of a compound assignment's property
    /// (`zend_std_get_property_ptr_ptr`, `Zend/zend_object_handlers.c`):
    /// an error handler run by the "Undefined property" warning may define
    /// the property, and the operation then works on that value
    /// (`zend_assign_op_overloaded_property`). `after` says how the
    /// operation goes on when that handler throws (see
    /// [`Vm::rw_dynamic_threw`]).
    ///
    /// `obj` is an [`Vm::obj_walk`] temporary.
    fn fetch_property_rw(&mut self, obj: &Obj, name: &ZStr, after: RwThrow<'_>) -> VmResult<Value> {
        match self.rw_create_dynamic(obj, name, 1) {
            Ok(Some(v)) => return Ok(v),
            Ok(None) => {}
            Err(t) => return self.rw_dynamic_threw(obj, name, after, t),
        }
        let v = self.read_property(obj, name, FetchMode::Read)?;
        if !matches!(v, Value::Null) || self.error_handlers.is_empty() {
            return Ok(v);
        }
        Ok(stored_property(obj, name).unwrap_or(v))
    }

    /// An error handler threw `t` at the dynamic-property deprecation or
    /// the "Undefined property" warning of a read-modify-write of the
    /// missing property `name`. `zend_std_get_property_ptr_ptr` adds it as
    /// null first (and keeps it), and the opcode goes on with the exception
    /// set: `++`/`--` still change it (`zend_pre_incdec_property_zval`; the
    /// exception is thrown once the result is written, see `RW_THROW`),
    /// and the other operators store their result (`concat_function`
    /// converts the null with `zval_try_get_string_func`, which does not
    /// fail).
    #[cold]
    #[inline(never)]
    fn rw_dynamic_threw(
        &mut self,
        obj: &Obj,
        name: &ZStr,
        after: RwThrow<'_>,
        t: Throw,
    ) -> VmResult<Value> {
        let cls = object::class_of(obj);
        let creatable = cls.flags
            & (crate::class::cflags::READONLY | crate::class::cflags::NO_DYNAMIC_PROPERTIES)
            == 0
            && !cls.is_enum();
        if !creatable || !matches!(t, Throw::Exception(_)) {
            return Err(t);
        }
        if stored_property(obj, name).is_none() {
            self.write_dynamic_prop(obj, name.clone(), Value::Null);
        }
        match after {
            RwThrow::IncDec => {
                RW_THROW.with(|c| c.set(Some(t)));
                Ok(Value::Null)
            }
            RwThrow::Op(op, rhs) => {
                if let Ok(nv) = self.assign_op_value(op, &Value::Null, rhs) {
                    self.write_dynamic_prop(obj, name.clone(), nv);
                }
                Err(t)
            }
        }
    }

    /// `place = $cv` where `$cv` is undefined. `ZEND_ASSIGN_DIM` and
    /// `ZEND_ASSIGN_OBJ` (`Zend/zend_vm_def.h`) read `OP_DATA` once they
    /// have the container (and for an array the slot), so its warning
    /// follows the fetches' diagnostics (an error handler that throws on
    /// them never sees it) and is not raised when the walk fails first:
    /// the walk raises it when it gets there ([`PendingRead`]).
    #[cold]
    fn place_assign_undef_cv_impl(&mut self, pid: PlaceId, cv: u32) -> VmResult<Value> {
        // `$o->{$p} = $v` / `$this->$p = $v`: ZEND_ASSIGN_OBJ fetches the
        // object, then reads OP_DATA, and only then its property name
        // (OP2), so the value's warning precedes the name's (bug #79599,
        // Zend/tests/gh10251.phpt); an error handler that throws on it
        // stops before the name is read. Resolving the place would read a
        // dynamic name first, so raise the value's warning before it.
        let value_first = {
            let p = &self.frame().code.places[pid as usize];
            matches!(p.base, PlaceBase::Cv(_) | PlaceBase::This)
                && matches!(
                    p.segs.as_slice(),
                    [Seg::Prop(
                        Operand::Cv(_) | Operand::Tmp(_) | Operand::TmpKeep(_)
                    )]
                )
        };
        if value_first {
            let name = self.frame().code.cv_names[cv as usize].clone();
            self.warning(&crate::exec::undefined_var_msg(&name))?;
            let (base, segs) = self.resolve_place_ex(pid, true, false, PlaceOp::Assign)?;
            return self.write_place(base, &segs, WOp::Assign(Value::Null));
        }
        let (base, mut segs) = self.resolve_place_ex(pid, true, false, PlaceOp::Assign)?;
        if segs.is_empty() {
            let v = self.read_cv(cv)?;
            return self.write_place(base, &segs, WOp::Assign(v));
        }
        let name = self.frame().code.cv_names[cv as usize].clone();
        segs.add_read(VALUE, name);
        self.write_place(base, &segs, WOp::Assign(Value::Null))
    }

    /// Raises the warning of the undefined variable pending as the value
    /// assigned by the walk over `segs` ([`Vm::place_assign_undef_cv`])
    /// before an object's final write (`offsetSet`, a native
    /// `write_dimension`, the property's), as `ZEND_ASSIGN_DIM` and
    /// `ZEND_ASSIGN_OBJ` read `OP_DATA` before calling the handlers.
    /// [`Vm::take_pending_value_warning`] for an opcode that goes on after
    /// an exception the error handler throws: that exception is returned
    /// (see [`Vm::rw_undefined_var`]); an exit or a fatal error stops at
    /// once.
    fn held_value_warning(&mut self, segs: &[RSeg]) -> VmResult<Option<Throw>> {
        match take_pending(segs, VALUE) {
            Some(name) => self.rw_undefined_var(&name),
            None => Ok(None),
        }
    }

    /// The last step of `$o[k] = v` or `$o[k] op= v` on an object with
    /// native dimension handlers (`ArrayObject`, `SplFixedArray`, ...)
    /// after the "Undefined variable" warning of the offset or the value
    /// threw `held`: php-src goes on with the opcode, so the handler
    /// still writes (`spl_array_write_dimension()` appends for a null
    /// offset), and an error it throws is chained onto `held`
    /// (`zend_throw_exception_internal()` while `EG(exception)` is set).
    /// A method the class overrides in user code is not called
    /// (`zend_call_function()` returns at once with an exception
    /// pending), and neither does the handler then write. Returns the
    /// exception to throw.
    #[cold]
    #[inline(never)]
    fn native_dim_write_held(
        &mut self,
        obj: &Obj,
        key: Option<&Value>,
        wop: &WOp,
        held: Throw,
    ) -> Throw {
        let Throw::Exception(prev) = &held else {
            return held;
        };
        let Some(h) = crate::ext::hooks_of(obj) else {
            return held;
        };
        let cls = object::class_of(obj).clone();
        let user = |name: &[u8]| {
            cls.find_method(name)
                .is_some_and(|m| matches!(m.func, crate::func::Func::User(_)))
        };
        let r = self.with_diagnostics_lost(|vm| match wop {
            WOp::Assign(v) if !user(b"offsetset") => match h.write_dimension {
                Some(f) => f(vm, obj, key, v).map(|_| ()),
                None => Ok(()),
            },
            WOp::AssignOp(op, rhs) if !user(b"offsetget") && !user(b"offsetset") => {
                // `zend_binary_assign_op_obj_dim`: read, operate, write.
                let k = key.cloned().unwrap_or(Value::Null);
                match (h.read_dimension, h.write_dimension) {
                    (Some(read), Some(write)) => match read(vm, obj, &k) {
                        Ok(Some(cur)) => match vm.assign_op_value(*op, &cur.deref(), rhs) {
                            Ok(nv) => {
                                drop(cur);
                                write(vm, obj, key, &nv).map(|_| ())
                            }
                            Err(e) => Err(e),
                        },
                        Ok(None) => Ok(()),
                        Err(e) => Err(e),
                    },
                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        });
        match r {
            Ok(()) => held,
            Err(Throw::Exception(new)) => {
                crate::exceptions::chain_previous(&new, prev);
                Throw::Exception(new)
            }
            Err(t) => t,
        }
    }

    /// Runs `f` as php-src runs code while an exception is pending: the
    /// user error handler is not called (`zend_call_function()` returns at
    /// once, and `zend_error_impl()` then reports nothing), so a
    /// diagnostic `f` raises is neither shown nor kept for
    /// `error_get_last()`. Errors outside the handler's mask take the
    /// default path as before; a fatal error still stops.
    #[cold]
    #[inline(never)]
    fn with_diagnostics_lost<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let mask = self
            .error_handlers
            .last()
            .filter(|h| !h.callback.is_null())
            .map_or(0, |h| h.mask);
        let reporting = self.error_reporting;
        self.error_reporting &= !mask;
        let saved = self
            .error_handlers
            .last_mut()
            .map(|h| std::mem::replace(&mut h.mask, 0));
        let last = self.last_error.clone();
        let r = f(self);
        let lost = self
            .last_error
            .as_ref()
            .is_some_and(|e| e.level & mask != 0);
        if lost {
            self.last_error = last;
        }
        self.error_reporting = reporting;
        if let (Some(m), Some(h)) = (saved, self.error_handlers.last_mut()) {
            h.mask = m;
        }
        r
    }

    /// The error of an array access on an object without ArrayAccess:
    /// `zend_bad_array_access` from `zend_std_write_dimension` and the
    /// other standard dimension handlers (Zend/zend_object_handlers.c),
    /// chained onto `held`, the exception already in flight, as
    /// `zend_throw_error` chains onto `EG(exception)`. `$o[k] op= v`
    /// throws twice (`zend_binary_assign_op_obj_dim`, Zend/zend_execute.c:
    /// `read_dimension` fails, then `zend_use_object_as_array`), the
    /// second error chained onto the first.
    #[cold]
    #[inline(never)]
    fn object_as_array_error(&mut self, obj: &Obj, held: Option<Throw>, assign_op: bool) -> Throw {
        let msg = format!(
            "Cannot use object of type {} as array",
            object::display_name(&object::class_of(obj).name)
        );
        let mut prev = match held {
            Some(Throw::Exception(o)) => Some(o),
            Some(t) => return t,
            None => None,
        };
        for _ in 0..usize::from(assign_op) {
            match self.make_throw("Error", &msg) {
                Throw::Exception(o) => {
                    if let Some(p) = &prev {
                        crate::exceptions::chain_previous(&o, p);
                    }
                    prev = Some(o);
                }
                t => return t,
            }
        }
        let t = self.make_throw("Error", &msg);
        if let (Throw::Exception(o), Some(p)) = (&t, &prev) {
            crate::exceptions::chain_previous(o, p);
        }
        t
    }

    fn take_pending_value_warning(&mut self, segs: &[RSeg]) -> VmResult<()> {
        match take_pending(segs, VALUE) {
            Some(name) => self.warning(&crate::exec::undefined_var_msg(&name)),
            None => Ok(()),
        }
    }

    /// `place op= v`
    /// `undef_value`: `v` is the undefined compiled variable read by the
    /// walk ([`Vm::place_assign_op_undef_cv`]).
    fn place_assign_op_impl(
        &mut self,
        pid: PlaceId,
        op: BinaryOp,
        mut v: Value,
        undef_value: Option<u32>,
    ) -> VmResult<Value> {
        let (single, head) = {
            let p = &self.frame().code.places[pid as usize];
            let fast = undef_value.is_none();
            (
                fast && propcache::cacheable(p),
                fast && propcache::cacheable_head(p),
            )
        };
        // Fast paths: ASSIGN_OBJ_OP on a cached property: `.=` on a string
        // appends in place, another operator on a number operates in place.
        if single {
            if op == BinaryOp::Concat {
                if let Some(r) = self.prop_cache_concat(pid, &v) {
                    return Ok(r);
                }
            }
            if let Some(r) = self.prop_cache_assign_op(pid, op, &v)? {
                return Ok(r);
            }
        } else if head {
            // ASSIGN_DIM_OP on an element of a cached array property.
            if let Some(r) = self.prop_cache_nested_assign_op(pid, op, &v) {
                return Ok(r);
            }
        }
        if let Some(cv) = self.rw_undef_cv_container(pid) {
            self.rw_undef_container_first(pid, cv)?;
        }
        let (base, mut segs) = self.resolve_place(pid, true)?;
        if matches!(
            self.frame().code.places[pid as usize].base,
            PlaceBase::Global(_)
        ) {
            self.rw_undefined_global(pid, &base)?;
        }
        if let Some(cv) = undef_value {
            if segs.is_empty() {
                v = self.read_cv(cv)?;
            } else {
                let name = self.frame().code.cv_names[cv as usize].clone();
                segs.add_read(VALUE, name);
            }
        }
        if !segs.is_empty() {
            // `$this->a[$k] op= v` & co.: enter a cached declared slot.
            if head && !single {
                if let Some((obj, idx)) = self.prop_cache_container(pid) {
                    drop(base);
                    return self.write_cached_container(&obj, idx, &segs, WOp::AssignOp(op, v));
                }
            }
            // One walk: offsets convert and object handlers run once.
            return self.write_place(base, &segs, WOp::AssignOp(op, v));
        }
        // `$a += $b` of two arrays: `write_var_root` makes the union in the
        // variable's own array.
        if let (BinaryOp::Add, Value::Array(_), RBase::Var(store, idx)) = (op, &v, &base) {
            if matches!(store.borrow().slots.get(*idx), Some(Some(Value::Array(_)))) {
                return self.write_place(base, &segs, WOp::AssignOp(op, v));
            }
        }
        // Fast path: a plain variable holding an int or float
        // (`ZEND_ASSIGN_OP` on a CV): no destructor, reference or type
        // constraint can be involved, so the slot is updated in place.
        if let RBase::Var(store, idx) = &base {
            let cur = match store.borrow().slots.get(*idx) {
                Some(Some(n @ (Value::Int(_) | Value::Float(_)))) => Some(n.clone()),
                _ => None,
            };
            if let Some(cur) = cur {
                let nv = ops::binary_op(op, &cur, &v, self).map_err(|e| self.rt_throw(e))?;
                if let Some(slot) = store.borrow_mut().slots.get_mut(*idx) {
                    *slot = Some(nv.clone());
                }
                self.frame_mut().sync_from_symbol_table();
                if self.store_in_globals(store) {
                    self.sync_global_frames();
                }
                return Ok(nv);
            }
        }
        // Fast path: plain variable holding a string being appended to.
        if let RBase::Var(store, idx) = &base {
            if op == BinaryOp::Concat {
                if let Value::Str(add) = &v {
                    self.clear_cvs_for_slot(store, *idx);
                    let mut st = store.borrow_mut();
                    if let Some(Value::Str(s)) = st.slots.get_mut(*idx).and_then(Option::as_mut) {
                        if !ZStr::ptr_eq(s, add) {
                            s.append_str(add);
                            let res = Value::Str(s.clone());
                            drop(st);
                            self.frame_mut().sync_from_symbol_table();
                            if self.store_in_globals(store) {
                                self.sync_global_frames();
                            }
                            return Ok(res);
                        }
                        // `$s .= $s`: php-src extends the one string in
                        // place (`concat_function`, result == op1 == op2),
                        // so the operand's handle goes first and the memory
                        // limit sees only the growth.
                        drop(st);
                        drop(v);
                        let mut st = store.borrow_mut();
                        let Some(Value::Str(s)) = &mut st.slots[*idx] else {
                            unreachable!("the slot held the string a moment ago")
                        };
                        s.append_self();
                        let res = Value::Str(s.clone());
                        drop(st);
                        self.frame_mut().sync_from_symbol_table();
                        if self.store_in_globals(store) {
                            self.sync_global_frames();
                        }
                        return Ok(res);
                    }
                    drop(st);
                    self.frame_mut().sync_from_symbol_table();
                    if self.store_in_globals(store) {
                        self.sync_global_frames();
                    }
                }
            }
        }
        let base2 = base.clone();
        // An undefined variable's warning may throw (see
        // `Vm::rw_undefined_var`): the result is stored first.
        let saved = RW_THROW.with(std::cell::Cell::take);
        let cur = self.write_place(base, &segs, WOp::FetchRw);
        let pending = RW_THROW.with(|c| c.replace(saved));
        let cur = cur?;
        let nv = match ops::binary_op(op, &cur, &v, self) {
            Ok(nv) => nv,
            Err(e) => {
                let err = self.rt_throw(e);
                let in_place = concats_in_place(op, &cur);
                drop(cur);
                return Err(match pending {
                    Some(t) => t,
                    None if in_place => err,
                    None => self.failed_assign_op_root(&base2, err),
                });
            }
        };
        drop(cur);
        let r = self.write_place(base2, &segs, WOp::Assign(nv))?;
        match pending {
            Some(t) => Err(t),
            None => Ok(r),
        }
    }

    /// The undefined compiled variable a read-modify-write of `pid` fetches
    /// as a dimension container (`$a[$k] op= v`, `$a[$k]++`).
    #[inline]
    fn rw_undef_cv_container(&self, pid: PlaceId) -> Option<u32> {
        let f = self.frame();
        let p = &f.code.places[pid as usize];
        let PlaceBase::Cv(cv) = p.base else {
            return None;
        };
        if !matches!(p.segs.first(), Some(Seg::Dim(_) | Seg::Append)) {
            return None;
        }
        let undef = f
            .cvs
            .get(cv as usize)
            .map_or(false, crate::vm::CvSlot::is_none);
        undef.then_some(cv)
    }

    /// `ZEND_ASSIGN_DIM_OP` / `ZEND_FETCH_DIM_RW` (`Zend/zend_vm_def.h`)
    /// fetch the container (`BP_VAR_RW`: "Undefined variable", then null)
    /// before reading the offset, so an undefined container warns before
    /// an undefined offset variable does.
    #[cold]
    #[inline(never)]
    fn rw_undef_container_first(&mut self, pid: PlaceId, cv: u32) -> VmResult<()> {
        let warned = self.read_cv(cv);
        // The fetch goes on after an error handler threw at the warning
        // (`EG(exception)` set): the undefined variable becomes an empty
        // array, and an illegal offset still throws, chained after it.
        let init = if warned.is_ok() {
            Value::Null
        } else {
            Value::Array(ZArray::new())
        };
        let f = self.frame_mut();
        let idx = cv as usize;
        if let Some(slot) = f.cvs.get_mut(idx) {
            if slot.is_none() {
                slot.set(Some(init.clone()));
                f.sync_cv_to_symtab(cv, &init);
            }
        }
        let Err(Throw::Exception(old)) = warned else {
            return warned.map(drop);
        };
        Err(self.rw_illegal_first_offset(pid, old))
    }

    /// The exception of [`Vm::rw_undef_container_first`] whose handler
    /// threw `old`: `zend_illegal_container_offset`'s `TypeError` for a
    /// literal, variable or temporary array or object first offset, chained after
    /// `old` (displayed as "Next TypeError"), else `old`.
    #[cold]
    fn rw_illegal_first_offset(&mut self, pid: PlaceId, old: Obj) -> Throw {
        let key = match self.frame().code.places[pid as usize].segs.first() {
            Some(Seg::Dim(Operand::Lit(i))) => Some(self.lit(*i)),
            Some(Seg::Dim(Operand::Cv(cv))) => self.cv_get(*cv),
            Some(Seg::Dim(Operand::Tmp(t) | Operand::TmpKeep(t))) => {
                self.frame().tmps.get(*t as usize).cloned()
            }
            _ => None,
        };
        let Some(key @ (Value::Array(_) | Value::Object(_))) = key.as_ref().map(Value::deref)
        else {
            return Throw::Exception(old);
        };
        let msg = format!(
            "Cannot access offset of type {} on array",
            crate::typecheck::given_type(&key)
        );
        match self.make_throw("TypeError", &msg) {
            Throw::Exception(new) => {
                crate::exceptions::chain_previous(&new, &old);
                Throw::Exception(new)
            }
            other => other,
        }
    }

    /// The read half of a read-modify-write: the current value, when the
    /// walk ended on an `ArrayAccess` offset that object and offset, and
    /// the exception an error handler threw at a missing dynamic property,
    /// to throw once the operation is done (see `RW_THROW`). `None` when
    /// the walk was abandoned because an error handler changed the array
    /// (see `rw_container_changed`, `Vm::live_give_up`).
    #[allow(clippy::type_complexity)] // (value, overloaded ArrayAccess target, pending throw)
    fn fetch_rw(
        &mut self,
        pid: PlaceId,
        base: RBase,
        segs: &[RSeg],
    ) -> VmResult<Option<(Value, Option<(Obj, Value)>, Option<Throw>)>> {
        match segs.last() {
            Some(RSeg::Dim(_) | RSeg::Append) => {}
            Some(RSeg::Prop(_)) => {
                // Saved and restored around the walk, whose user code (an
                // error handler) may run increments of its own.
                let saved = RW_THROW.with(std::cell::Cell::take);
                let r = self.write_place_cached(pid, base, segs, WOp::FetchRw);
                let thrown = RW_THROW.with(|c| c.replace(saved));
                return Ok(Some((r?, None, thrown)));
            }
            None => {
                // An undefined variable's warning may throw (see
                // `Vm::rw_undefined_var`).
                let saved = RW_THROW.with(std::cell::Cell::take);
                let r = self.write_place_cached(pid, base, segs, WOp::FetchRw);
                let thrown = RW_THROW.with(|c| c.replace(saved));
                return Ok(Some((r?, None, thrown)));
            }
        }
        // Save and restore around the walk: user code it runs (offsetGet,
        // destructors, error handlers) may perform read-modify-writes of
        // its own.
        let saved = OVERLOADED_RW.with(std::cell::Cell::take);
        let saved_skip = RW_SKIPPED.with(|c| c.replace(false));
        let r = self.write_place_cached(pid, base, segs, WOp::FetchRw);
        let mine = OVERLOADED_RW.with(|c| c.replace(saved));
        let skipped = RW_SKIPPED.with(|c| c.replace(saved_skip));
        let r = r?;
        Ok((!skipped).then_some((r, mine, None)))
    }

    /// `$cv .= <string or int>` (`ZEND_ASSIGN_OP` on a CV, whose
    /// `concat_function` call has `result == op1`): a compiled variable
    /// holding a string (not a reference) is appended to in place, which
    /// extends the buffer when nothing else holds it (`zend_string_extend`),
    /// without resolving the place generically. Writes the result to `dst`
    /// and returns `true`; `false` leaves the operation to
    /// [`Vm::place_assign_op`], whose string fast path does the same after
    /// the generic resolution (integers it converts to a temporary string
    /// first).
    #[inline]
    pub(crate) fn concat_assign_cv(&mut self, pid: PlaceId, v: &Value, dst: Option<u32>) -> bool {
        let Some(result) = concat_cv_in_place(self.frame(), pid, v, dst.is_some()) else {
            return false;
        };
        if let Some(t) = dst {
            self.set_tmp(t, result);
        }
        true
    }

    /// [`Vm::concat_assign_cv`] for a literal right operand (`$s .= "x"`,
    /// `ZEND_ASSIGN_OP` CV, CONST), appended straight from the literal
    /// table without reading (cloning) the operand. `false` when `val` is
    /// not a literal or the fast path does not apply; the caller then
    /// reads the operand as usual.
    #[inline]
    pub(crate) fn concat_assign_cv_lit(
        &mut self,
        pid: PlaceId,
        val: Operand,
        dst: Option<u32>,
    ) -> bool {
        let Operand::Lit(l) = val else {
            return false;
        };
        let f = self.frame();
        let Some(result) = concat_cv_in_place(f, pid, &f.code.literals[l as usize], dst.is_some())
        else {
            return false;
        };
        if let Some(t) = dst {
            self.set_tmp(t, result);
        }
        true
    }

    /// The int fast path of [`Vm::place_incdec`] before the generic place
    /// resolution: a plain compiled variable holding an int directly.
    #[inline]
    fn incdec_cv_int_fast(&mut self, pid: PlaceId, kind: IncDecKind) -> Option<Value> {
        let f = self.frame_mut();
        let p = &f.code.places[pid as usize];
        let (PlaceBase::Cv(cv), true) = (&p.base, p.segs.is_empty()) else {
            return None;
        };
        let cv = *cv;
        let idx = cv as usize;
        match f.cvs.get(idx)?.get() {
            Some(Value::Int(i)) => {
                let mut int_val = *i;
                let res = incdec_int_slot(&mut int_val, kind)?;
                f.cvs[idx].set(Some(Value::Int(int_val)));
                f.sync_cv_to_symtab(cv, &Value::Int(int_val));
                Some(res)
            }
            _ => None,
        }
    }

    /// `++place` etc.
    #[allow(clippy::too_many_lines)]
    fn place_incdec_impl(&mut self, pid: PlaceId, kind: IncDecKind) -> VmResult<Value> {
        // Fast path: `$i++` on a compiled variable holding an int (no
        // overflow), before resolving the place generically.
        if let Some(r) = self.incdec_cv_int_fast(pid, kind) {
            return Ok(r);
        }
        // Fast paths: PRE/POST_INC/DEC_OBJ on a cached property, and the
        // element of a cached array property (`$this->a[$k]++`).
        let (single, head, stat) = {
            let p = &self.frame().code.places[pid as usize];
            (
                propcache::cacheable(p),
                propcache::cacheable_head(p),
                p.segs.is_empty()
                    && matches!(
                        p.base,
                        PlaceBase::StaticProp {
                            name: Operand::Lit(_),
                            ..
                        }
                    ),
            )
        };
        // `self::$n++` / `A::$n--`: the static property's cached cell
        // (`ZEND_*_INC_STATIC_PROP` with `CACHED_PTR`).
        if stat {
            if let Some(r) = self.inc_dec_static_prop_fast(pid, kind) {
                return Ok(r);
            }
        }
        if single {
            if let Some(r) = self.prop_cache_incdec(pid, kind) {
                return Ok(r);
            }
        } else if head {
            if let Some(r) = self.prop_cache_nested_incdec(pid, kind) {
                return Ok(r);
            }
        }
        if let Some(cv) = self.rw_undef_cv_container(pid) {
            self.rw_undef_container_first(pid, cv)?;
        }
        let (base, segs) = self.resolve_place(pid, true)?;
        if matches!(
            self.frame().code.places[pid as usize].base,
            PlaceBase::Global(_)
        ) {
            self.rw_undefined_global(pid, &base)?;
        }
        let base2 = match &base {
            RBase::Var(s, i) => RBase::Var(s.clone(), *i),
            RBase::This(o) => RBase::This(o.clone()),
            RBase::Cell(c) => RBase::Cell(c.clone()),
            RBase::Value(v) => RBase::Value(v.clone()),
            RBase::Globals => RBase::Globals,
            RBase::Undef(n) => RBase::Undef(n.clone()),
        };
        // Fast path for integer variables.
        if segs.is_empty() {
            if let Some(r) = incdec_var_int_fast(&base, kind) {
                self.frame_mut().sync_from_symbol_table();
                if self.base_in_globals(&base) {
                    self.sync_global_frames();
                }
                return Ok(r);
            }
        }
        let Some((cur, overloaded, thrown)) = self.fetch_rw(pid, base, &segs)? else {
            // FETCH_DIM_RW yielded a null temporary: ++/-- act on it alone.
            let mut nv = Value::Null;
            if kind.is_inc() {
                ops::increment(&mut nv, self).map_err(|e| self.rt_throw(e))?;
            } else {
                ops::decrement(&mut nv, self).map_err(|e| self.rt_throw(e))?;
            }
            return Ok(if kind.is_post() { Value::Null } else { nv });
        };
        // `FETCH_DIM_RW` on an object: a non-reference result is a
        // temporary, so the increment is not written back.
        let temporary = match &overloaded {
            Some((obj, _)) if !matches!(cur, Value::Reference(_)) => {
                if !matches!(cur, Value::Object(_)) {
                    self.notice(&format!(
                        "Indirect modification of overloaded element of {} has no effect",
                        object::class_of(obj).name
                    ))?;
                }
                true
            }
            _ => false,
        };
        // Objects: property inc/dec of undefined, string offsets checked.
        let mut nv = cur.deref();
        let r = if kind.is_inc() {
            ops::increment(&mut nv, self)
        } else {
            ops::decrement(&mut nv, self)
        };
        if let Err(e) = r {
            let err = self.rt_throw(e);
            return Err(self.incdec_write_back(base2, &segs, nv, err));
        }
        let old = cur.deref();
        if temporary {
            return Ok(if kind.is_post() { old } else { nv });
        }
        let guard = crate::typed_refs::IncDecGuard::new(kind.is_inc(), &old, &nv);
        let res = if let Value::Reference(r) = &cur {
            self.assign_ref(r, nv)
        } else {
            // A silent write-back walk: `FETCH_DIM_RW` converted the
            // offsets (and reported them) once already, also those of the
            // containers reached through properties (`QUIET_AT`).
            let through_props = segs.iter().any(|s| matches!(s, RSeg::Prop(_)));
            let quiet =
                through_props.then(|| QUIET_AT.with(|q| q.replace(Some(self.frames.len()))));
            let r = if let Some((obj, idx)) = self.cached_container_of(pid, &segs) {
                drop(base2);
                self.write_cached_container(&obj, idx, &segs, WOp::Assign(nv))
            } else {
                let private = self.cv_walk_private(&base2);
                let globals = self.base_in_globals(&base2);
                let r = self.write_place_inner(base2, &segs, WOp::Assign(nv), true);
                flush_deferred_ref_assigns();
                if !private {
                    self.frame_mut().sync_from_symbol_table();
                    if globals {
                        self.sync_global_frames();
                    }
                }
                r
            };
            if let Some(q) = quiet {
                QUIET_AT.with(|c| c.set(q));
            }
            r
        };
        drop(guard);
        let res = res?;
        if let Some(t) = thrown {
            return Err(t);
        }
        Ok(if kind.is_post() { old } else { res })
    }

    /// `zend_{pre,post}_incdec_overloaded_property`: a property that is
    /// not modified in place (readonly, hooked, magic, asymmetric
    /// visibility) is written back even when the increment failed, so the
    /// write's own error ("Cannot modify readonly property ...") becomes
    /// the exception, with the `TypeError` as its previous one. Writing an
    /// ordinary property back is unobservable.
    #[cold]
    fn incdec_write_back(&mut self, base: RBase, segs: &[RSeg], old: Value, err: Throw) -> Throw {
        if !matches!(segs.last(), Some(RSeg::Prop(_))) {
            return err;
        }
        match self.write_place(base, segs, WOp::Assign(old)) {
            Ok(_) => err,
            Err(Throw::Exception(new)) => {
                if let Throw::Exception(prev) = &err {
                    crate::exceptions::chain_previous(&new, prev);
                }
                Throw::Exception(new)
            }
            Err(t) => t,
        }
    }

    /// `place =& v`
    fn place_assign_ref_impl(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        let v = match v {
            Value::Reference(r) => Value::Reference(r),
            other => {
                let segs = &self.frame().code.places[pid as usize].segs;
                let to_prop = matches!(segs.last(), Some(Seg::Prop(_)));
                let to_dim = !segs.is_empty() && !segs.iter().any(|s| matches!(s, Seg::Prop(_)));
                if to_prop {
                    return self.place_assign_ref_nonvar_prop(pid, other);
                }
                if to_dim {
                    return self.assign_non_ref_to_dim(pid, other);
                }
                self.notice("Only variables should be assigned by reference")?;
                return self.place_assign(pid, other);
            }
        };
        let (base, segs) = self.resolve_place(pid, true)?;
        self.write_place(base, &segs, WOp::AssignRef(v))
    }

    /// `$a[...] =& f()` where `f()` did not return a reference:
    /// `FETCH_DIM_W` runs first, so its errors ("Cannot add element to the
    /// array as the next element is already occupied") take precedence;
    /// then `zend_wrong_assign_to_variable_reference()` raises the notice
    /// and assigns the value to the fetched element.
    #[cold]
    #[inline(never)]
    fn assign_non_ref_to_dim(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        let r = self.place_make_ref(pid)?;
        self.notice("Only variables should be assigned by reference")?;
        self.assign_ref(&r, v)
    }

    /// `$o->p =& f()` where `f()` returned a value: `ZEND_ASSIGN_OBJ_REF`
    /// (`zend_assign_to_property_reference`, `Zend/zend_execute.c`) fetches
    /// the property for writing before it looks at the value, so `__get()`
    /// runs first and a value it hands out throws "Cannot assign by
    /// reference to overloaded object" instead of the notice (bug69732).
    /// The walk carries the value as a fresh reference and marks itself
    /// for [`Vm::assign_property_ref`] (see [`take_nonvar_ref`]).
    #[cold]
    #[inline(never)]
    fn place_assign_ref_nonvar_prop(&mut self, pid: PlaceId, v: Value) -> VmResult<Value> {
        let (base, segs) = self.resolve_place(pid, true)?;
        let depth = self.frames.len();
        let saved = PENDING_NONVAR_REF.with(|c| c.replace(Some(depth)));
        let r = self.write_place(
            base,
            &segs,
            WOp::AssignRef(Value::Reference(PhpRef::new(v))),
        );
        PENDING_NONVAR_REF.with(|c| c.set(saved));
        r
    }

    /// Makes a place a reference.
    fn place_make_ref_impl(&mut self, pid: PlaceId) -> VmResult<PhpRef> {
        // Fast path: `&$this->p` on a cached declared slot.
        if propcache::cacheable(&self.frame().code.places[pid as usize]) {
            if let Some(rf) = self.prop_cache_make_ref(pid) {
                return Ok(rf);
            }
        }
        let (base, segs) = self.resolve_place(pid, true)?;
        // `&$this->a[$k]` & co. enter a cached declared slot.
        match self.write_place_cached(pid, base, &segs, WOp::MakeRef)? {
            Value::Reference(r) => Ok(r),
            other => Ok(PhpRef::new(other)),
        }
    }

    /// `unset(place)`
    fn place_unset_impl(&mut self, pid: PlaceId) -> VmResult<()> {
        let code = self.frame().code.clone();
        let p = &code.places[pid as usize];
        if let (PlaceBase::StaticProp { class, name }, true) = (&p.base, p.segs.is_empty()) {
            return Err(self.unset_static_prop(class, *name));
        }
        // Fast path: UNSET_OBJ of a cached property.
        if propcache::cacheable(p) && self.prop_cache_unset(pid) {
            return Ok(());
        }
        let (base, segs) = match self.resolve_place_ex(pid, false, false, PlaceOp::Unset) {
            Ok(r) => r,
            Err(t @ Throw::Exception(_)) => return Err(self.unset_after_offset_threw(pid, t)),
            Err(t) => return Err(t),
        };
        if let RBase::Undef(_) = &base {
            return Ok(());
        }
        if let RBase::Var(store, idx) = &base {
            // UNSET_DIM, FETCH_DIM_UNSET and FETCH_OBJ_UNSET on an
            // undefined CV warn (ZVAL_UNDEFINED_OP1); a lone UNSET_OBJ
            // does not.
            let warns = !segs.is_empty() && !matches!(&segs[..], [RSeg::Prop(_)]);
            if warns
                && store.borrow().slots[*idx].is_none()
                && matches!(
                    self.frame().code.places[pid as usize].base,
                    PlaceBase::Cv(_)
                )
            {
                let name = store.borrow().names[*idx].clone();
                return self.warning(&crate::exec::undefined_var_msg(&name));
            }
        }
        if segs.is_empty() {
            if let RBase::Cell(_) = base {
                return Err(self.make_throw("Error", "Attempt to unset static property"));
            }
            if let RBase::This(_) = base {
                return Ok(());
            }
        }
        self.write_place(base, &segs, WOp::Unset).map(|_| ())
    }

    /// `unset($obj[$undef])` whose "Undefined variable" warning was turned
    /// into an exception by the error handler: `ZEND_UNSET_DIM` goes on
    /// with a null offset, and for an object without ArrayAccess
    /// `zend_std_unset_dimension` throws "Cannot use object of type ... as
    /// array" (`zend_bad_array_access`), chained onto the handler's
    /// exception, as for the writes ([`Vm::object_as_array_error`]). Any
    /// other place gives the exception back as it is.
    #[cold]
    #[inline(never)]
    fn unset_after_offset_threw(&mut self, pid: PlaceId, t: Throw) -> Throw {
        let code = self.frame().code.clone();
        let Some(p) = code.places.get(pid as usize) else {
            return t;
        };
        let (PlaceBase::Cv(cv), [Seg::Dim(_)]) = (&p.base, &p.segs[..]) else {
            return t;
        };
        let Some(Value::Object(obj)) = self.cv_get(*cv).map(|v| v.deref()) else {
            return t;
        };
        if crate::ext::hooks_of(&obj).is_some() || self.instance_of(&obj, b"arrayaccess") {
            return t;
        }
        self.object_as_array_error(&obj, Some(t), false)
    }

    /// `ZEND_UNSET_STATIC_PROP`: the class is fetched, then the name, and
    /// `zend_std_unset_static_property` always throws.
    #[cold]
    fn unset_static_prop(&mut self, class: &crate::opcode::ClassSpec, name: Operand) -> Throw {
        let cls = match self.resolve_class_spec(class) {
            Ok(c) => c,
            Err(t) => return t,
        };
        let n = match self.read(name).and_then(|v| self.to_zstr(&v)) {
            Ok(n) => n,
            Err(t) => return t,
        };
        self.make_throw(
            "Error",
            &format!(
                "Attempt to unset static property {}::${}",
                object::display_name(&cls.name),
                n.to_string_lossy()
            ),
        )
    }

    /// The read warning for an undefined variable base: `$GLOBALS[...]`
    /// fetches carry `ZEND_FETCH_GLOBAL` ("Undefined global variable",
    /// `zend_fetch_var_address_helper`).
    #[cold]
    fn undefined_place_var_msg(&self, pid: PlaceId, name: &ZStr) -> String {
        if matches!(
            self.frame().code.places[pid as usize].base,
            PlaceBase::Global(_)
        ) {
            format!("Undefined global variable ${name}")
        } else {
            crate::exec::undefined_var_msg(name)
        }
    }

    /// Reads a place.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn place_fetch(&mut self, pid: PlaceId, mode: FetchMode) -> VmResult<Value> {
        // `$this->p[$k][$j]` & co. on an array property: no walk.
        if let Some(v) = self.fetch_prop_dims_fast(pid, mode) {
            return Ok(v);
        }
        // `$GLOBALS['name'][$k]` on a variable of the global scope.
        if let Some(v) = self.fetch_global_dims_fast(pid, mode) {
            return Ok(v);
        }
        if matches!(
            self.frame().code.places[pid as usize].base,
            PlaceBase::Global(_) | PlaceBase::GlobalsArray
        ) {
            self.attach_globals_symbol_table();
        }
        if mode == FetchMode::Read {
            let direct_auto_global = {
                let p = &self.frame().code.places[pid as usize];
                if let (PlaceBase::Global(Operand::Lit(lit)), true) = (&p.base, p.segs.is_empty()) {
                    match self.frame().code.literals.get(*lit as usize) {
                        Some(Value::Str(s)) if s.as_bytes() == b"_GET" => Some(0),
                        Some(Value::Str(s)) if s.as_bytes().first() == Some(&b'_') => {
                            crate::auto_globals::index(s.as_bytes())
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            };
            if let Some(idx) = direct_auto_global {
                let val = self.get_auto_global(idx);
                if val.is_null() {
                    let is_unset = if let Some(slot) = self.ext.auto_globals.cached_slots[idx] {
                        self.globals
                            .borrow()
                            .slots
                            .get(slot as usize)
                            .map_or(true, |s| s.is_none())
                    } else {
                        true
                    };
                    if is_unset {
                        let name = ZStr::from(crate::auto_globals::AUTO_GLOBAL_NAMES[idx]);
                        let msg = self.undefined_place_var_msg(pid, &name);
                        self.warning(&msg)?;
                        return Ok(Value::Null);
                    }
                }
                return Ok(val);
            }
        }
        let head = {
            let p = &self.frame().code.places[pid as usize];
            // Fast path: reading a plain compiled variable (`f($x)`).
            if mode == FetchMode::Read {
                if let (PlaceBase::Cv(cv), true) = (&p.base, p.segs.is_empty()) {
                    let cv = *cv;
                    return self.read_cv(cv);
                }
                // FETCH_DIM_FUNC_ARG sent by value: `$a[]` fails before its
                // container or any later offset is read
                // (`zend_use_undef_in_read_context`).
                if matches!(p.segs.first(), Some(Seg::Append)) {
                    return Err(self.make_throw("Error", "Cannot use [] for reading"));
                }
            }
            // Fast path: `$cv->name` / `$this->name` on a cached property
            // (FETCH_OBJ_R/IS with a runtime cache slot); a longer chain
            // (`$this->a[$i]`, `$this->a->b`) goes on from that value.
            if mode != FetchMode::List && propcache::cacheable_head(p) {
                let single = p.segs.len() == 1;
                // `$this->a[$k]`: the element straight from the cached slot.
                if !single {
                    if let Some(value) = self.prop_cache_read_dim(pid) {
                        return Ok(value);
                    }
                }
                match self.prop_cache_read(pid) {
                    Some(value) if single => return Ok(value),
                    found => found,
                }
            } else {
                None
            }
        };
        if let Some(v) = self.fetch_global_fast(pid, mode) {
            return Ok(v);
        }
        let code = self.frame().code.clone();
        let place = &code.places[pid as usize];
        let segs = &place.segs;
        let (mut cur, rest) = if let Some(v) = head {
            (v, &segs[1..])
        } else if matches!(place.base, PlaceBase::GlobalsArray) && !segs.is_empty() {
            let Seg::Dim(dim_op) = &segs[0] else {
                return Err(self.make_throw("Error", "Cannot use [] for reading"));
            };
            let k = self.read_dim_operand(*dim_op)?;
            let v = self.fetch_global_by_key(&k, mode)?;
            (v, &segs[1..])
        } else {
            let base = match self.cv_read_base(&place.base) {
                Some(b) => b,
                None => self.resolve_base(&place.base, false, mode == FetchMode::Is, None)?,
            };
            (self.fetch_base_value(pid, base, segs, mode)?, &segs[..])
        };
        // Each offset/name operand is read when its FETCH_DIM_*/FETCH_OBJ_*
        // runs (after the container's warnings); a quiet chain that reached
        // null still reads, and reports, the remaining operands.
        let mut settled = mode == FetchMode::Is && rest.len() < segs.len() && cur.is_null();
        let cv_base = rest.len() == segs.len() && matches!(place.base, PlaceBase::Cv(_));
        for (i, seg) in rest.iter().enumerate() {
            match seg {
                Seg::Dim(op) => {
                    let held = if cv_base && i == 0 {
                        shared_handles(&cur)
                    } else {
                        0
                    };
                    let k = self.read_dim_operand(*op)?;
                    if held > 1 && shared_handles(&cur) == 1 {
                        // php-src reads the variable through its slot once the
                        // offset is read: an error handler of the offset's
                        // warning that released its value leaves what the
                        // variable holds now.
                        let base = match self.cv_read_base(&place.base) {
                            Some(b) => b,
                            None => {
                                self.resolve_base(&place.base, false, mode == FetchMode::Is, None)?
                            }
                        };
                        cur = self.fetch_base_value(pid, base, segs, mode)?;
                    }
                    if !settled {
                        cur = self.fetch_dim(cur, &k, mode)?;
                    }
                }
                Seg::Append => {
                    return Err(self.make_throw("Error", "Cannot use [] for reading"));
                }
                Seg::Prop(op) => {
                    let v = self.read(*op)?;
                    if !settled {
                        cur = match cur {
                            Value::Object(o) => {
                                // The last segment's own runtime cache entry.
                                let hit = self.prop_cache_read_tail(pid, &o, *op, mode);
                                match hit {
                                    Some(found) => found,
                                    None => {
                                        let name = self.to_zstr(&v)?;
                                        self.read_property(&o, &name, mode)?
                                    }
                                }
                            }
                            other => {
                                if mode == FetchMode::Read {
                                    let name = self.to_zstr(&v)?;
                                    self.warning(&format!(
                                        "Attempt to read property \"{}\" on {}",
                                        name.to_string_lossy(),
                                        other.value_name()
                                    ))?;
                                }
                                Value::Null
                            }
                        };
                    }
                }
            }
            if mode == FetchMode::Is && cur.is_null() {
                settled = true;
            }
        }
        Ok(cur)
    }

    /// The value of a read place's resolved base (`place_fetch` without a
    /// cached head), with the undefined-variable warnings of a read.
    fn fetch_base_value(
        &mut self,
        pid: PlaceId,
        base: RBase,
        segs: &[Seg],
        mode: FetchMode,
    ) -> VmResult<Value> {
        Ok(match base {
            RBase::Var(store, idx) => {
                let v = store.borrow().slots[idx].as_ref().map(Value::deref);
                match v {
                    Some(v) => v,
                    None => {
                        if mode != FetchMode::Is {
                            let name = store.borrow().names[idx].clone();
                            let msg = self.undefined_place_var_msg(pid, &name);
                            self.warning(&msg)?;
                        }
                        Value::Null
                    }
                }
            }
            RBase::This(o) => match o {
                Some(o) => Value::Object(o),
                None => {
                    // FETCH_OBJ_IS / FETCH_DIM_IS on `$this` still need it.
                    if mode == FetchMode::Is && segs.is_empty() {
                        Value::Null
                    } else {
                        return Err(
                            self.make_throw("Error", "Using $this when not in object context")
                        );
                    }
                }
            },
            RBase::Cell(c) => {
                self.typed_static_read_check(&c, mode)?;
                c.borrow().deref()
            }
            RBase::Value(v) => v.deref(),
            RBase::Undef(name) => {
                if mode != FetchMode::Is {
                    let msg = self.undefined_place_var_msg(pid, &name);
                    self.warning(&msg)?;
                }
                Value::Null
            }
            RBase::Globals => self.globals_array(),
        })
    }

    /// One dimension read (`zend_fetch_dimension_address_read`).
    pub(crate) fn fetch_dim(&mut self, cur: Value, k: &Value, mode: FetchMode) -> VmResult<Value> {
        match cur {
            Value::Array(a) => {
                let mut diags = Vec::new();
                // FETCH_DIM_IS (`??`, the containers of a nested isset())
                // converts like a read: only the final ISSET_ISEMPTY_DIM_OBJ
                // says "in isset or empty".
                let key = match key_of(k, &mut diags, KeyCtx::Read) {
                    Ok(k) => k,
                    Err(e) => {
                        self.flush_diags(diags)?;
                        return Err(self.walk_err(e));
                    }
                };
                self.flush_diags(diags)?;
                match a.get(&key) {
                    Some(v) => Ok(v.deref()),
                    None => {
                        if mode != FetchMode::Is {
                            self.warning(&undefined_key_msg(&key))?;
                        }
                        Ok(Value::Null)
                    }
                }
            }
            Value::Str(s) => {
                if mode == FetchMode::List {
                    // zend_fetch_dimension_address_read with is_list: a
                    // string is not destructured.
                    self.warning("Cannot use string as array")?;
                    return Ok(Value::Null);
                }
                let mut diags = Vec::new();
                let off = if mode == FetchMode::Is {
                    string_offset_is(k, &mut diags)
                } else {
                    string_offset(k, &mut diags, false).map(Some)
                };
                let warned = !diags.is_empty();
                let held = s.refcount();
                self.flush_diags(diags)?;
                // php-src holds a reference to the string while it warns:
                // when an error handler released all the others, the string
                // is freed and the result is null.
                if warned && held > 1 && s.refcount() == 1 {
                    return Ok(Value::Null);
                }
                let off = match off {
                    Ok(Some(o)) => o,
                    Ok(None) => return Ok(Value::Null),
                    Err(e) => return Err(self.walk_err(e)),
                };
                let len = i64::try_from(s.len()).unwrap_or(i64::MAX);
                let real = if off < 0 { off + len } else { off };
                if real < 0 || real >= len {
                    if mode == FetchMode::Is {
                        return Ok(Value::Null);
                    }
                    self.warning(&format!("Uninitialized string offset {off}"))?;
                    return Ok(Value::Str(ZStr::empty()));
                }
                Ok(Value::Str(ZStr::from_byte(
                    s.as_bytes()[usize::try_from(real).unwrap_or(0)],
                )))
            }
            Value::Object(o) => {
                if let Some(h) = crate::ext::hooks_of(&o) {
                    if mode == FetchMode::Is && !h.read_dimension_is {
                        if let Some(f) = h.has_dimension {
                            if let Some(false) = f(self, &o, k, false)? {
                                return Ok(Value::Null);
                            }
                        }
                    }
                    if let Some(f) = h.read_dimension {
                        self.dim_ctx = if mode == FetchMode::Is {
                            crate::ext::DimensionCtx::Is
                        } else {
                            crate::ext::DimensionCtx::Read
                        };
                        let r = f(self, &o, k);
                        self.dim_ctx = crate::ext::DimensionCtx::Read;
                        if let Some(v) = r? {
                            return Ok(v.deref());
                        }
                    }
                }
                if !self.instance_of(&o, b"arrayaccess") {
                    return Err(self.make_throw(
                        "Error",
                        &format!(
                            "Cannot use object of type {} as array",
                            object::display_name(&object::class_of(&o).name)
                        ),
                    ));
                }
                if mode == FetchMode::Is {
                    let e = self.call_method_name_n(&o, "offsetExists", [k.clone()])?;
                    if !self.to_bool(&e)? {
                        return Ok(Value::Null);
                    }
                }
                self.call_method_name_n(&o, "offsetGet", [k.clone()])
            }
            Value::Null => {
                if mode == FetchMode::Read {
                    self.warning("Trying to access array offset on null")?;
                }
                Ok(Value::Null)
            }
            other => {
                match mode {
                    FetchMode::Read => {
                        self.warning(&format!(
                            "Trying to access array offset on {}",
                            other.value_name()
                        ))?;
                    }
                    FetchMode::List => {
                        self.warning(&format!("Cannot use {} as array", other.type_name()))?;
                    }
                    FetchMode::Is => {}
                }
                Ok(Value::Null)
            }
        }
    }

    /// `isset(place)` / `empty(place)` (returns the isset or empty result).
    #[allow(clippy::too_many_lines)]
    /// `isset($GLOBALS[$name])` with the name in a variable or a literal
    /// string (`ZEND_ISSET_ISEMPTY_VAR` on the global symbol table): the
    /// variable's value straight from the table
    /// ([`Vm::fetch_global_by_key_optional`]), without the place walk
    /// (MediaWiki's `GlobalVarConfig` asks this for every setting it
    /// reads). `None` for `empty()`, any other place or name, and the
    /// name of an auto-global.
    fn isset_global_fast(&mut self, pid: PlaceId, empty: bool) -> Option<bool> {
        if empty {
            return None;
        }
        let f = self.frame();
        let p = f.code.places.get(pid as usize)?;
        let PlaceBase::Global(name) = p.base else {
            return None;
        };
        if !p.segs.is_empty() {
            return None;
        }
        let k = match name {
            Operand::Cv(cv) => f.cvs.get(cv as usize).and_then(crate::vm::CvSlot::get)?,
            Operand::Lit(l) => f.code.literals.get(l as usize)?,
            _ => return None,
        };
        let Value::Str(name) = k else {
            return None;
        };
        // An auto-global's name is the walk's: `isset()` does not create
        // one that is still armed (`auto_globals_jit`), which the fetch
        // below would.
        let bytes = name.as_bytes();
        if bytes.first() == Some(&b'_') && crate::auto_globals::index(bytes).is_some() {
            return None;
        }
        let k = k.clone();
        Some(
            self.fetch_global_by_key_optional(&k)
                .is_some_and(|v| !v.is_null()),
        )
    }

    pub(crate) fn place_isset(&mut self, pid: PlaceId, empty: bool) -> VmResult<bool> {
        // `isset($this->p[$k][$j])` & co. on an array property: no walk.
        if let Some(r) = self.isset_prop_dims_fast(pid, empty) {
            return Ok(r);
        }
        if let Some(r) = self.isset_global_fast(pid, empty) {
            return Ok(r);
        }
        let (single, head) = {
            let p = &self.frame().code.places[pid as usize];
            (propcache::cacheable(p), propcache::cacheable_head(p))
        };
        // Fast paths: ISSET_ISEMPTY_PROP_OBJ on a cached property, and
        // ISSET_ISEMPTY_DIM_OBJ on a cached array property's element.
        if single {
            if let Some(r) = self.prop_cache_isset(pid, empty)? {
                return Ok(r);
            }
        } else if head {
            if let Some(r) = self.prop_cache_nested_isset(pid, empty)? {
                return Ok(r);
            }
        }
        // `isset($this->a[$k])` & co.: the container's FETCH_OBJ_IS from
        // the cache.
        let head = if head && !single {
            self.prop_cache_read(pid)
        } else {
            None
        };
        let code = self.frame().code.clone();
        let place = &code.places[pid as usize];
        if matches!(place.base, PlaceBase::Global(_) | PlaceBase::GlobalsArray) {
            self.attach_globals_symbol_table();
        }
        let segs = &place.segs;
        let (mut cur, start) = if let Some(v) = head {
            (Some(v), 1)
        } else if matches!(place.base, PlaceBase::GlobalsArray) && !segs.is_empty() {
            self.attach_globals_symbol_table();
            let Seg::Dim(dim_op) = &segs[0] else {
                return Ok(empty);
            };
            let k = self.read_dim_operand(*dim_op)?;
            let v = self.fetch_global_by_key_optional(&k);
            (v, 1)
        } else {
            let base = match self.cv_read_base(&place.base) {
                Some(b) => b,
                None => self.resolve_base(&place.base, false, true, None)?,
            };
            let cur = match base {
                RBase::Var(store, idx) => store.borrow().slots[idx].as_ref().map(Value::deref),
                // ZEND_ISSET_ISEMPTY_PROP_OBJ / FETCH_*_IS on an unset `$this`
                // (zend_this_not_in_object_context_helper).
                RBase::This(None) if !segs.is_empty() => {
                    return Err(self.make_throw("Error", "Using $this when not in object context"));
                }
                RBase::This(o) => o.map(Value::Object),
                RBase::Cell(c) => {
                    (!crate::typed_refs::static_cell_uninit(&c)).then(|| c.borrow().deref())
                }
                RBase::Value(v) => Some(v.deref()),
                RBase::Undef(_) => None,
                RBase::Globals => Some(self.globals_array()),
            };
            (cur, 0)
        };
        let n = segs.len();
        if n == 0 {
            return Ok(match cur {
                None => empty,
                Some(v) => {
                    if empty {
                        !self.to_bool(&v)?
                    } else {
                        !v.is_null()
                    }
                }
            });
        }
        // Each operand is read when its opcode runs, also once the result
        // is known (an undefined offset variable still warns); property
        // names convert only when an object is reached.
        let mut settled = false;
        for (i, seg) in segs.iter().enumerate().skip(start) {
            let raw = match seg {
                Seg::Dim(op) => self.read_dim_operand(*op)?,
                Seg::Prop(op) => self.read(*op)?,
                Seg::Append => Value::Null,
            };
            if settled {
                continue;
            }
            let c = match cur.take() {
                Some(c) if !c.is_null() => c,
                _ => {
                    settled = true;
                    continue;
                }
            };
            let last = i + 1 == n;
            if !last {
                let v = match seg {
                    Seg::Dim(_) => self.fetch_dim(c, &raw, FetchMode::Is)?,
                    Seg::Prop(_) => match c {
                        Value::Object(o) => {
                            let name = self.to_zstr(&raw)?;
                            self.read_property(&o, &name, FetchMode::Is)?
                        }
                        _ => Value::Null,
                    },
                    Seg::Append => {
                        return Err(self.make_throw("Error", "Cannot use [] for reading"))
                    }
                };
                cur = Some(v);
                continue;
            }
            // Last segment.
            let k = &raw;
            return match seg {
                Seg::Dim(_) => match c {
                    Value::Array(a) => {
                        // The lookup snapshot is Zend's temporary GC addref,
                        // not an owner that survives a diagnostic callback.
                        let guarded = matches!(k, Value::Float(_) | Value::Null);
                        let had_owner = guarded && a.refcount() > 1;
                        let mut diags = Vec::new();
                        let key = match key_of(k, &mut diags, KeyCtx::Isset) {
                            Ok(k) => k,
                            Err(e) => return Err(self.walk_err(e)),
                        };
                        // isset() does not emit the null-offset deprecation twice.
                        self.flush_diags(diags)?;
                        if had_owner && a.refcount() == 1 {
                            return Ok(empty);
                        }
                        match a.get(&key) {
                            None => Ok(empty),
                            Some(v) => {
                                let v = v.deref();
                                if empty {
                                    Ok(!self.to_bool(&v)?)
                                } else {
                                    Ok(!v.is_null())
                                }
                            }
                        }
                    }
                    Value::Str(s) => {
                        let off = match k {
                            Value::Int(i) => Some(*i),
                            Value::Str(ks) => match php_runtime::numeric::is_numeric_string_ex(
                                ks.as_bytes(),
                                false,
                            ) {
                                Some(n) => match n.value {
                                    php_runtime::numeric::Number::Int(i) => Some(i),
                                    php_runtime::numeric::Number::Float(_) => None,
                                },
                                None => None,
                            },
                            Value::Float(d) => Some(self.isset_float_string_offset(*d)?),
                            Value::Bool(b) => Some(i64::from(*b)),
                            Value::Null => Some(0),
                            _ => None,
                        };
                        let len = i64::try_from(s.len()).unwrap_or(0);
                        match off {
                            Some(o) => {
                                let real = if o < 0 { o + len } else { o };
                                if real < 0 || real >= len {
                                    Ok(empty)
                                } else if empty {
                                    Ok(s.as_bytes()[usize::try_from(real).unwrap_or(0)] == b'0')
                                } else {
                                    Ok(true)
                                }
                            }
                            None => Ok(empty),
                        }
                    }
                    Value::Object(o) => {
                        if let Some(f) = crate::ext::hooks_of(&o).and_then(|h| h.has_dimension) {
                            if let Some(r) = f(self, &o, k, empty)? {
                                return Ok(if empty { !r } else { r });
                            }
                        }
                        if !self.instance_of(&o, b"arrayaccess") {
                            return Err(self.make_throw(
                                "Error",
                                &format!(
                                    "Cannot use object of type {} as array",
                                    object::display_name(&object::class_of(&o).name)
                                ),
                            ));
                        }
                        let e = self.call_method_name_n(&o, "offsetExists", [k.clone()])?;
                        let exists = self.to_bool(&e)?;
                        if !empty {
                            return Ok(exists);
                        }
                        if !exists {
                            return Ok(true);
                        }
                        let v = self.call_method_name_n(&o, "offsetGet", [k.clone()])?;
                        Ok(!self.to_bool(&v)?)
                    }
                    _ => Ok(empty),
                },
                Seg::Prop(_) => match c {
                    Value::Object(o) => {
                        let name = self.to_zstr(k)?;
                        self.has_property(&o, &name, empty)
                    }
                    _ => Ok(empty),
                },
                Seg::Append => Err(self.make_throw("Error", "Cannot use [] for reading")),
            };
        }
        Ok(empty)
    }

    /// `zval_get_long_ex(offset, /* is_strict */ true)` on a float offset
    /// of a string in `isset()/empty()` (`zend_isset_dim_slow`): the
    /// not-representable warning, then the precision-loss deprecation.
    #[cold]
    fn isset_float_string_offset(&mut self, d: f64) -> VmResult<i64> {
        let l = php_runtime::convert::dval_to_lval_silent(d);
        let shown = php_runtime::float::format_double(d, -1, false);
        if !d.is_finite() || !crate::typecheck::fits_long(d) {
            self.warning(&format!(
                "The float {shown} is not representable as an int, cast occurred"
            ))?;
        }
        if !php_runtime::convert::is_long_compatible(d, l) {
            self.error(
                E_DEPRECATED,
                &format!("Implicit conversion from float {shown} to int loses precision"),
            )?;
        }
        Ok(l)
    }

    /// Binding a global variable: ensures the main frame's symbol table is attached
    /// and returns a reference to the global slot.
    #[allow(dead_code)]
    pub(crate) fn op_bind_global(&mut self, name: &[u8]) -> PhpRef {
        self.attach_globals_symbol_table();
        self.global_ref(name)
    }

    /// Global read: ensures the main frame's symbol table is attached and fetches
    /// the global variable by key.
    #[allow(dead_code)]
    pub(crate) fn fetch_global_read(&mut self, k: &Value, mode: FetchMode) -> VmResult<Value> {
        self.attach_globals_symbol_table();
        self.fetch_global_by_key(k, mode)
    }

    /// Looks up a global variable by key value without rebuilding the global symbol table.
    pub(crate) fn fetch_global_by_key_optional(&mut self, k: &Value) -> Option<Value> {
        self.attach_globals_symbol_table();
        let name_bytes: std::borrow::Cow<[u8]> = match k {
            Value::Str(s) => std::borrow::Cow::Borrowed(s.as_bytes()),
            Value::Int(i) => std::borrow::Cow::Owned(i.to_string().into_bytes()),
            _ => return None,
        };
        if is_superglobal_name(&name_bytes) {
            self.is_auto_global(&name_bytes);
        }
        // The main frame's own variable when that frame is the newest of
        // the global scope; while a top-level include or `eval` runs above
        // it (`Vm::main_is_global_top`), the table, which
        // `attach_globals_symbol_table` just brought up to date from that
        // frame, has the current value.
        let main = self
            .frames
            .first()
            .filter(|f| f.kind == crate::vm::FrameKind::Main && self.main_is_global_top());
        if let Some(main_frame) = main {
            if let Some(cv) = main_frame
                .code
                .cv_names
                .iter()
                .position(|n| n.as_bytes() == &name_bytes[..])
            {
                if let Some(slot) = main_frame.cvs.get(cv) {
                    if let Some(v) = slot.get() {
                        return Some(v.deref());
                    }
                }
                return None;
            }
        }
        let idx = self.globals.borrow_mut().find(&name_bytes)?;
        let g = self.globals.borrow();
        g.slots.get(idx as usize)?.as_ref().map(Value::deref)
    }

    /// Looks up a global variable by key value, emitting a warning if undefined (in non-Is mode).
    pub(crate) fn fetch_global_by_key(&mut self, k: &Value, mode: FetchMode) -> VmResult<Value> {
        self.attach_globals_symbol_table();
        if let Some(v) = self.fetch_global_by_key_optional(k) {
            return Ok(v);
        }
        if mode != FetchMode::Is {
            let name_str = match k {
                Value::Str(s) => s.clone(),
                Value::Int(i) => ZStr::from(i.to_string()),
                other => self.to_zstr(other)?,
            };
            let msg = format!("Undefined global variable ${name_str}");
            self.warning(&msg)?;
        }
        Ok(Value::Null)
    }

    /// A reference to a global variable (`global $x`).
    pub(crate) fn global_ref(&mut self, name: &[u8]) -> PhpRef {
        self.attach_globals_symbol_table();
        let g = self.globals.clone();
        let idx = g.borrow_mut().find_or_add(name) as usize;
        self.global_slot_ref(&g, idx)
    }

    /// [`Vm::global_ref`] for a `global $name` statement, whose name is a
    /// literal of the op array: the slot found is remembered by the
    /// literal's address. A hit is checked against the name the slot has
    /// (the global table's slots never move while it lives; a reused
    /// address or a new table fails the check).
    pub(crate) fn global_ref_lit(&mut self, name: &ZStr) -> PhpRef {
        self.attach_globals_symbol_table();
        let g = self.globals.clone();
        let bytes = name.as_bytes();
        let addr = bytes.as_ptr() as usize;
        let e = (addr >> 3) & (GLOBAL_SLOT_CACHE - 1);
        let (ca, ci) = self.global_slot_cache[e];
        let idx = {
            let mut st = g.borrow_mut();
            if ca == addr
                && st
                    .names
                    .get(ci as usize)
                    .is_some_and(|n| n.as_bytes() == bytes)
                && (ci as usize) < st.slots.len()
            {
                ci as usize
            } else {
                let i = st.find_or_add(bytes);
                self.global_slot_cache[e] = (addr, i);
                i as usize
            }
        };
        self.global_slot_ref(&g, idx)
    }

    /// The reference global slot `idx` of `g` becomes (or already is). The
    /// global-scope frames are brought up to date only when the slot
    /// changed (a new variable, or a value that just became a reference).
    fn global_slot_ref(&mut self, g: &Rc<RefCell<VarStore>>, idx: usize) -> PhpRef {
        let mut st = g.borrow_mut();
        let Some(slot) = st.slots.get_mut(idx) else {
            unreachable!("find_or_add sized the table");
        };
        let was_ref = matches!(slot, Some(Value::Reference(_)));
        let r = slot.get_or_insert(Value::Null).make_ref();
        drop(st);
        if !was_ref {
            self.sync_global_frames();
        }
        r
    }
}

pub(crate) fn is_superglobal_name(n: &[u8]) -> bool {
    matches!(
        n,
        b"_GET"
            | b"_POST"
            | b"_COOKIE"
            | b"_SERVER"
            | b"_ENV"
            | b"_REQUEST"
            | b"_FILES"
            | b"_SESSION"
    )
}

/// Where a property's value lives for nested writes.
pub(crate) enum PropContainer {
    /// A declared property slot.
    Declared(usize),
    /// A dynamic property.
    Dynamic(ArrayKey),
    /// A value returned by `__get` (modifications are lost).
    Value(Value),
}

/// The value stored for `$obj->name` (a declared slot without hooks, or a
/// dynamic property), bypassing magic; `None` when unset or when `__get()`
/// or a lazy initializer would be involved.
fn stored_property(obj: &Obj, name: &ZStr) -> Option<Value> {
    let d = object::data(obj);
    if d.class.magic.get.is_some() || d.lazy.borrow().is_some() {
        return None;
    }
    match d.class.prop_map.get(name) {
        Some(pi) if pi.hooks.is_none() => d
            .slots
            .borrow()
            .get(pi.slot as usize)
            .and_then(|s| s.as_ref().map(Value::deref)),
        Some(_) => None,
        None => d
            .dynamic
            .borrow()
            .as_ref()
            .and_then(|t| t.get(&ArrayKey::Str(name.clone())).map(Value::deref)),
    }
}

pub(crate) mod propcache;

/// The array key operand `k` of frame `f` stands for, when reading it can
/// report nothing: an int or a string literal, or a compiled variable
/// holding one (not through a reference). `None` otherwise.
fn quiet_key(f: &crate::vm::Frame, k: Operand) -> Option<ArrayKey> {
    let v = match k {
        Operand::Lit(i) => f.code.literals.get(i as usize)?,
        Operand::Cv(cv) => f.cvs.get(cv as usize)?.get()?,
        _ => return None,
    };
    match v {
        Value::Int(i) => Some(ArrayKey::Int(*i)),
        Value::Str(s) => Some(ArrayKey::from_zstr(s)),
        _ => None,
    }
}

/// `ZEND_FETCH_DIM_W` + make-ref of element `key` of array `a`: the array
/// separated when shared, the element created as null when missing, and
/// the element made a reference (or its reference returned).
fn element_make_ref(a: &mut ZArray, key: ArrayKey) -> PhpRef {
    let t = a.make_mut();
    if let Some(v) = t.get_mut(&key) {
        return v.make_ref();
    }
    let r = PhpRef::new(Value::Null);
    t.insert(key, Value::Reference(r.clone()));
    r
}

#[cfg(test)]
mod tests;
