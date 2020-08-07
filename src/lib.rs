//! # `redox-buffer-pool`
//!
//! This crate provides a buffer pool for general-purpose memory management, with support for
//! allocating slices within the pool, as well as expanding the pool with potentially non-adjacent
//! larger underlying memory allocations, like _mmap(2)_ or other larger possible page-sized
//! allocations.
//!
//! The current allocator uses one B-trees to partition the space into regions either marked as
//! occupied or free. The keys used by the B-tree have a custom comparator, which ensures that keys
//! for used ranges are orderered after the keys for free ranges. This, together with having a free
//! space tree, makes acquiring buffer slices possible in O(log n), provided that there is already
//! an aligned range (otherwise, it simply does linear search in O(n) until it finds a range large
//! enough to account for the misalignment).

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![feature(option_expect_none, option_unwrap_none, map_first_last)]
#![cfg_attr(test, feature(slice_fill, vec_into_raw_parts))]

use core::borrow::{Borrow, BorrowMut};
use core::convert::{TryFrom, TryInto};
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{cmp, fmt, mem, ops, ptr, slice};

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::{Arc, Weak};

#[cfg(any(test, feature = "std"))]
use parking_lot::{RwLock, RwLockUpgradableReadGuard};

#[cfg(not(any(test, feature = "std")))]
use spinning::{RwLock, RwLockUpgradableReadGuard};

mod private {
    use core::{fmt, ops};

    pub trait Sealed {}
    pub trait IntegerRequirements:
        Sized
        + From<u8>
        + Copy
        + Clone
        + fmt::Debug
        + fmt::Display
        + Eq
        + PartialEq<Self>
        + PartialOrd<Self>
        + Ord
        + From<u8>
        + ops::Add<Self, Output = Self>
        + ops::AddAssign
        + ops::Sub<Self, Output = Self>
        + ops::Shl<u8, Output = Self>
        + ops::Shl<u32, Output = Self>
        + ops::Shr<u8, Output = Self>
        + ops::Shr<u32, Output = Self>
        + ops::Not<Output = Self>
        + ops::BitAnd<Output = Self>
        + ops::BitAndAssign
        + ops::BitOr<Output = Self>
        + ops::BitOrAssign
        + ops::BitXor<Self, Output = Self>
    {
        fn zero() -> Self {
            Self::from(0u8)
        }
        const MAX: Self;
        fn trailing_zeros(self) -> u32;

        fn try_into_usize(self) -> Option<usize>;
        fn checked_add(self, rhs: Self) -> Option<Self>;
        fn checked_sub(self, rhs: Self) -> Option<Self>;
        fn checked_div(self, rhs: Self) -> Option<Self>;
        fn checked_mul(self, rhs: Self) -> Option<Self>;
        fn is_power_of_two(self) -> bool;
    }
}

/// A type that can be used as offsets and lengths within a buffer pool. The default integer is
/// u32.
pub unsafe trait Integer: private::Sealed + private::IntegerRequirements {}
fn occ_map_ready_shift<I: Integer>() -> u32 {
    let bit_count = (mem::size_of::<I>() * 8) as u32;
    bit_count - 1
}
fn occ_map_used_bit<I: Integer>() -> I {
    I::from(1u8) << occ_map_ready_shift::<I>()
}
fn occ_map_off_mask<I: Integer>() -> I {
    !occ_map_used_bit::<I>()
}

macro_rules! impl_integer_for_primitive(
    ($primitive:ident) => {
        impl private::IntegerRequirements for $primitive {
            fn trailing_zeros(self) -> u32 {
                Self::trailing_zeros(self)
            }
            const MAX: Self = Self::MAX;

            fn try_into_usize(self) -> Option<usize> {
                usize::try_from(self).ok()
            }
            fn checked_add(self, rhs: Self) -> Option<Self> {
                Self::checked_add(self, rhs)
            }
            fn checked_sub(self, rhs: Self) -> Option<Self> {
                Self::checked_add(self, rhs)
            }
            fn checked_div(self, rhs: Self) -> Option<Self> {
                Self::checked_div(self, rhs)
            }
            fn checked_mul(self, rhs: Self) -> Option<Self> {
                Self::checked_mul(self, rhs)
            }
            fn is_power_of_two(self) -> bool {
                Self::is_power_of_two(self)
            }
        }
        unsafe impl Integer for $primitive {}
    }
);

impl_integer_for_primitive!(u16);
impl_integer_for_primitive!(u32);
impl_integer_for_primitive!(u64);
impl_integer_for_primitive!(u128);
impl_integer_for_primitive!(usize);

impl private::Sealed for u16 {}
impl private::Sealed for u32 {}
impl private::Sealed for u64 {}
impl private::Sealed for u128 {}
impl private::Sealed for usize {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialOrd, PartialEq)]
struct Offset<I>(I);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialOrd, PartialEq)]
struct Size<I>(I);

impl<I: Integer> Size<I> {
    fn size(&self) -> I {
        self.0
    }
    fn from_size(size: I) -> Self {
        Self(size)
    }
}

// A key of the free space B-tree, storing the size and the alignment (which is computed based on
// the offset, which is the value of that tree).
//
// Obviously this key is comparable; it first compares the size, and then the alignment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FreeEntry<I> {
    size: I,
    offset: I,
}

impl<I: Integer> FreeEntry<I> {
    fn size(&self) -> I {
        self.size
    }
    fn offset(&self) -> I {
        self.offset
    }
    fn log2_of_alignment(&self) -> I {
        I::from(self.offset.trailing_zeros() as u8)
    }
    fn set_size(&mut self, size: I) {
        self.size = size;
    }
    fn set_offset(&mut self, offset: I) {
        self.offset = offset;
    }
    fn from_size_offset(size: I, offset: I) -> Self {
        Self { size, offset }
    }
}
impl<I: Integer> Ord for FreeEntry<I> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.size(), &other.size())
            .then(Ord::cmp(
                &self.log2_of_alignment(),
                &other.log2_of_alignment(),
            ))
            .then(Ord::cmp(&self.offset, &other.offset))
    }
}
impl<I: Integer> PartialOrd<Self> for FreeEntry<I> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct OccOffset<I>(I);

impl<I: Integer> OccOffset<I> {
    fn offset(&self) -> I {
        self.0 & occ_map_off_mask::<I>()
    }
    fn is_used(&self) -> bool {
        self.0 & occ_map_used_bit::<I>() == occ_map_used_bit::<I>()
    }
    fn set_offset(&mut self, offset: I) {
        assert_eq!(offset & occ_map_off_mask::<I>(), offset);
        self.0 &= !occ_map_off_mask::<I>();
        self.0 |= offset;
    }
    fn set_used(&mut self, used: bool) {
        self.0 &= !occ_map_used_bit::<I>();
        if used {
            self.0 |= occ_map_used_bit::<I>();
        }
    }
    fn from_offset_used(offset: I, used: bool) -> Self {
        let mut this = Self(I::zero());
        this.set_offset(offset);
        this.set_used(used);
        this
    }
}

impl<I: Integer> Ord for OccOffset<I> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.is_used(), &other.is_used()).then(Ord::cmp(&self.offset(), &other.offset()))
    }
}
impl<I: Integer> PartialOrd for OccOffset<I> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

impl<I: Integer, E: Copy> MmapInfo<I, E> {
    fn null() -> Self {
        Self {
            addr: Addr::Uninitialized,
            extra: MaybeUninit::uninit(),
            size: Size::from_size(I::zero()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MmapInfo<I, E: Copy> {
    size: Size<I>,
    extra: MaybeUninit<E>,
    addr: Addr,
}
#[derive(Clone, Copy, Debug)]
enum Addr {
    Initialized(NonNull<u8>),
    Uninitialized,
}

impl Addr {
    fn as_ptr(&self) -> *mut u8 {
        match self {
            Self::Initialized(ptr) => ptr.as_ptr(),
            Self::Uninitialized => ptr::null_mut(),
        }
    }
}

impl<I, E: Copy> MmapInfo<I, E> {
    fn is_init(&self) -> bool {
        matches!(self.addr, Addr::Initialized(_))
    }
}

/// Various options used by the buffer pool mainly to limit the range of possible sizes and
/// alignments.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferPoolOptions<I> {
    log2_minimum_alignment: u8,
    log2_maximum_alignment: u8,

    minimum_size: I,
    maximum_size: I,
}
impl<I: Integer> BufferPoolOptions<I> {
    /// Use the default buffer pool options.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the maximum possible alignment, causing allocations with larger alignments to fail
    /// immediately. This will also override the minimum possible alignment, if this would happen
    /// to be smaller than that.
    ///
    /// # Panics
    ///
    /// This method will panic if the alignment is not a valid power of two, or if it's zero.
    pub fn with_maximum_alignment(mut self, alignment: I) -> Self {
        assert!(alignment.is_power_of_two());
        assert_ne!(alignment, I::zero());

        let log2_align = alignment.trailing_zeros();
        self.log2_maximum_alignment = log2_align.try_into().unwrap();
        self.log2_minimum_alignment =
            cmp::min(self.log2_minimum_alignment, self.log2_maximum_alignment);
        self
    }
    /// Reduce the minimum alignment to 1.
    pub fn with_no_minimum_alignment(self) -> Self {
        self.with_minimum_alignment(I::from(1u8))
    }
    /// Set the minimum possible alignment, causing allocations with smaller alignments to use this
    /// alignment instead. This will override the maximum alignment, if this were to be larger than
    /// that.
    ///
    /// # Panics
    /// This method will panic if the alignment is not a power of two, or if it's zero.
    pub fn with_minimum_alignment(mut self, alignment: I) -> Self {
        assert!(alignment.is_power_of_two());
        assert_ne!(alignment, I::zero());

        let log2_align = alignment.trailing_zeros();
        self.log2_minimum_alignment = log2_align.try_into().unwrap();
        self.log2_maximum_alignment =
            cmp::max(self.log2_maximum_alignment, self.log2_maximum_alignment);

        self
    }
    /// Allow all possible alignments when allocating.
    pub fn with_no_maximum_alignment(self) -> Self {
        self.with_maximum_alignment(I::from((mem::size_of::<I>() * 8 - 1) as u8))
    }
    /// Set the maximum size that allocations can have.
    pub fn with_maximum_size(mut self, size: I) -> Self {
        self.maximum_size = size;
        self.minimum_size = cmp::min(self.minimum_size, self.maximum_size);
        self
    }
    /// Set the minimum size that allocations can have. While this will not affect the lengths of
    /// the buffer slices, it will round their capacities up to this number, giving them extra
    /// space that they can optionally expand to.
    pub fn with_minimum_size(mut self, size: I) -> Self {
        self.minimum_size = size;
        self.maximum_size = cmp::max(self.minimum_size, self.maximum_size);
        self
    }
}

impl<I: Integer> Default for BufferPoolOptions<I> {
    fn default() -> Self {
        let log2_minimum_alignment = mem::align_of::<usize>() as u8;
        let log2_maximum_alignment = (mem::size_of::<I>() * 8 - 1) as u8;

        Self {
            // Default to system alignment for usize.
            log2_minimum_alignment,
            // Default to unlimited. TODO
            log2_maximum_alignment: cmp::max(log2_minimum_alignment, log2_maximum_alignment),
            maximum_size: I::MAX,
            minimum_size: I::from(1u8),
        }
    }
}

/// A buffer pool, featuring a general-purpose 32-bit allocator, and slice guards.
// TODO: Expand doc
pub struct BufferPool<I: Integer, H: Handle<I, E>, E: Copy> {
    handle: Option<H>,

    options: BufferPoolOptions<I>,

    guarded_occ_count: AtomicUsize,

    //
    // TODO: Concurrent B-trees. Well, at least something more efficient that uses less
    // coarse-grained locks.
    //

    // Occupied entries, mapped offset + used => size.
    occ_map: RwLock<BTreeMap<OccOffset<I>, Size<I>>>,
    // Free entries containing size+align+offset, in that order.
    free_map: RwLock<BTreeSet<FreeEntry<I>>>,
    // "mmap" map, mapped offset => info. These aren't required to come from the mmap syscall; they
    // are general-purpose larger allocations that this buffer pool builds on. Hence, these can
    // also be io_uring "pool" shared memory or physalloc+physmap DMA allocations.
    mmap_map: RwLock<BTreeMap<Offset<I>, MmapInfo<I, E>>>,
}
unsafe impl<I, H, E> Send for BufferPool<I, H, E>
where
    I: Integer + Send,
    H: Handle<I, E> + Send,
    E: Copy + Send,
{
}
unsafe impl<I, H, E> Sync for BufferPool<I, H, E>
where
    I: Integer + Sync,
    H: Handle<I, E> + Sync,
    E: Copy + Sync,
{
}

impl<I, H, E> fmt::Debug for BufferPool<I, H, E>
where
    I: Integer,
    H: fmt::Debug + Handle<I, E>,
    E: Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("handle", &self.handle)
            // TODO: maps
            .finish()
    }
}

/// A no-op guard, that cannot be initialized but still useful in type contexts..
pub enum NoGuard {}
impl Guard for NoGuard {
    fn try_release(&self) -> bool {
        unreachable!("NoGuard cannot be initialized")
    }
}

/// A handle type that cannot be initialized, causing the handle to take up no space in the buffer
/// pool struct.
pub enum NoHandle {}

impl<I, E> Handle<I, E> for NoHandle
where
    E: Copy,
{
    type Error = ::core::convert::Infallible;

    fn close(&mut self, _entries: MmapEntries<I, E>) -> Result<(), Self::Error> {
        unreachable!("NoHandle cannot be initialized")
    }
}

// TODO: Support mutable/immutable slices, maybe even with refcounts? A refcount of 1 would mean
// exclusive, while a higher refcount would mean shared.
#[derive(Debug)]
/// A slice from the buffer pool, that can be read from or written to as a regular smart pointer.
pub struct BufferSlice<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard = NoGuard> {
    alloc_start: I,
    alloc_capacity: I,
    alloc_len: I,

    mmap_start: I,
    mmap_size: I,
    pointer: *mut u8,
    extra: E,

    pool: PoolRefKind<'a, I, H, E>,
    guard: Option<G>,
}

#[derive(Debug)]
enum PoolRefKind<'a, I: Integer, H: Handle<I, E>, E: Copy> {
    Ref(&'a BufferPool<I, H, E>),
    Strong(Arc<BufferPool<I, H, E>>),
    Weak(Weak<BufferPool<I, H, E>>),
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy> Clone for PoolRefKind<'a, I, H, E> {
    fn clone(&self) -> Self {
        match *self {
            Self::Ref(r) => Self::Ref(r),
            Self::Strong(ref arc) => Self::Strong(Arc::clone(arc)),
            Self::Weak(ref weak) => Self::Weak(Weak::clone(weak)),
        }
    }
}

unsafe impl<'a, I, H, E, G> Send for BufferSlice<'a, I, H, E, G>
where
    I: Integer,
    H: Handle<I, E> + Send + Sync,
    E: Copy + Send,
    G: Guard + Send,
{
}
unsafe impl<'a, I, H, E, G> Sync for BufferSlice<'a, I, H, E, G>
where
    I: Integer,
    H: Send + Sync + Handle<I, E>,
    E: Copy + Sync,
    G: Sync + Guard,
{
}

impl<'a, I, H, E, G> BufferSlice<'a, I, H, E, G>
where
    I: Integer,
    H: Handle<I, E>,
    E: Copy,
    G: Guard,
{
    /// Checks whether the pool that owns this slice is still alive, or if it has dropped. Note
    /// that this only makes sense for weak buffer slices, since buffer slices tied to a lifetime
    /// cannot outlive their pools (checked for at compile time), while strong buffer slices ensure
    /// at runtime that they outlive their pools.
    ///
    /// For weak buffer slices, this method should be called before doing anything with the slice,
    /// since a single deref could make it panic if the buffer isn't there anymore.
    pub fn pool_is_alive(&self) -> bool {
        match self.pool {
            PoolRefKind::Weak(ref w) => w.strong_count() > 0,
            PoolRefKind::Ref(_) | PoolRefKind::Strong(_) => true,
        }
    }

    /// Construct an immutable slice from this buffer.
    ///
    /// # Panics
    ///
    /// This method will panic if this is a weak slice, and the buffer pool has been destroyed..
    pub fn as_slice(&self) -> &[u8] {
        assert!(self.pool_is_alive());
        debug_assert!(self.alloc_capacity >= self.alloc_len);
        unsafe {
            slice::from_raw_parts(
                self.pointer as *const u8,
                self.alloc_len
                    .try_into_usize()
                    .expect("the buffer pool integer type is too large to fit within the system pointer width"),
            )
        }
    }
    /// Tries to construct an immutable slice from this buffer, returning None if the pool has been
    /// dropped (and hence, this is a weak slice).
    pub fn try_as_slice(&self) -> Option<&[u8]> {
        if self.pool_is_alive() {
            Some(self.as_slice())
        } else {
            None
        }
    }
    /// Construct a mutable slice from this buffer.
    ///
    /// # Panics
    ///
    /// Like [`as_slice`], this method will panic if the buffer pool has been destroyed.
    ///
    /// [`as_slice`]: #method.as_slice
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        assert!(self.pool_is_alive());
        debug_assert!(self.alloc_capacity >= self.alloc_len);
        unsafe {
            slice::from_raw_parts_mut(self.pointer, self.alloc_len.try_into_usize().expect("the buffer pool integer type is too large to fit within the system pointer width"))
        }
    }
    /// Tries to construct a mutable slice from this buffer, returning None if the pool has been
    /// destroyed.
    pub fn try_as_slice_mut(&mut self) -> Option<&mut [u8]> {
        if self.pool_is_alive() {
            Some(self.as_slice_mut())
        } else {
            None
        }
    }
    /// Forcefully remove a guard from a future, from this slice, returning it if there was a guard
    /// already.
    ///
    /// # Safety
    /// This is unsafe because it allows removing guards set by pending futures; although this is
    /// completely fine when there are no pending ones, the buffer slice will be reclaimed without
    /// the guard, causing UB if any producer keeps using its pointer.
    pub unsafe fn unguard(&mut self) -> Option<G> {
        match self.guard.take() {
            Some(g) => {
                let arc;

                let pool = match self.pool {
                    PoolRefKind::Ref(reference) => reference,
                    PoolRefKind::Strong(ref arc) => &*arc,
                    PoolRefKind::Weak(ref weak) => {
                        arc = weak.upgrade().expect(
                            "calling unguard on a weakly-owned buffer slice where the pool died",
                        );
                        &*arc
                    }
                };
                let prev = pool.guarded_occ_count.fetch_sub(1, Ordering::Release);
                assert_ne!(prev, 0, "someone forgot to increment the guarded_occ_count, now I'm getting a subtraction overflow!");
                Some(g)
            }
            None => None,
        }
    }
    /// Adds a guard to this buffer, preventing it from deallocating unless the guard accepts that.
    /// This is crucial when memory is shared with another component that may be outside this
    /// process's address space. If there is a pending io_uring submission or a pending NVME
    /// command for instance, this guard will fail if the buffer is in use by a command, and leak
    /// the memory instead when dropping.
    ///
    /// This will error with [`WithGuardError`] if there is already an active guard.
    pub fn guard(&mut self, guard: G) -> Result<(), WithGuardError<G>> {
        if self.guard.is_some() {
            return Err(WithGuardError { this: guard });
        }
        self.guard = Some(guard);

        let arc;

        let pool = match self.pool {
            PoolRefKind::Ref(pool) => pool,
            PoolRefKind::Strong(ref arc) => &*arc,
            PoolRefKind::Weak(ref pool_weak) => {
                arc = pool_weak.upgrade().expect(
                    "trying to guard weakly-owned buffer slice which pool has been dropped",
                );
                &*arc
            }
        };
        // TODO: Is Relaxed ok here?
        pool.guarded_occ_count.fetch_add(1, Ordering::Release);

        Ok(())
    }
    /// Tries to add a guard of potentially a different type than the guard type in this slice.
    /// Because of that this, this will consume self and construct a different `BufferSlice` with
    /// a different guard type, or error with `self` if there was already a guard present.
    pub fn with_guard<OtherGuard: Guard>(
        self,
        other: OtherGuard,
    ) -> Result<BufferSlice<'a, I, H, E, OtherGuard>, WithGuardError<Self>> {
        if self.guard.is_some() {
            return Err(WithGuardError { this: self });
        }
        let alloc_start = self.alloc_start;
        let alloc_capacity = self.alloc_capacity;
        let alloc_len = self.alloc_len;
        let mmap_start = self.mmap_start;
        let mmap_size = self.mmap_size;
        let pointer = self.pointer;
        let pool = self.pool.clone();
        let extra = self.extra;

        mem::forget(self);

        let mut slice = BufferSlice {
            alloc_start,
            alloc_capacity,
            alloc_len,
            mmap_start,
            mmap_size,
            pointer,
            pool,
            extra,
            guard: None,
        };
        slice.guard(other).unwrap();
        Ok(slice)
    }
    fn reclaim_inner(&mut self) -> bool {
        let arc;

        let pool = match self.pool {
            PoolRefKind::Ref(reference) => reference,
            PoolRefKind::Strong(ref aliased_arc) => {
                arc = Arc::clone(aliased_arc);
                &*arc
            }
            PoolRefKind::Weak(ref weak) => {
                arc = match weak.upgrade() {
                    Some(a) => a,
                    None => return true,
                };
                &*arc
            }
        };
        let (was_guarded, can_be_reclaimed) = match self.guard {
            Some(ref guard) => (true, guard.try_release()),
            None => (false, true),
        };
        if can_be_reclaimed {
            unsafe {
                if was_guarded {
                    self.unguard().unwrap();
                }
                pool.reclaim_slice_inner(&*self);
            }
            true
        } else {
            false
        }
    }
    /// Reclaim the buffer slice, equivalent to dropping but with a Result. If the buffer slice was
    /// guarded by a future, this will fail with [`ReclaimError`] if the future hadn't completed when
    /// this was called.
    pub fn reclaim(mut self) -> Result<(), ReclaimError<Self>> {
        match self.reclaim_inner() {
            true => {
                mem::forget(self);
                Ok(())
            }
            false => Err(ReclaimError { this: self }),
        }
    }

    /// Get the offset of the buffer pool where this was allocated.
    pub fn offset(&self) -> I {
        self.alloc_start
    }
    /// Get the length of the allocation slice.
    pub fn len(&self) -> I {
        self.alloc_len
    }
    /// Get the capacity of the allocation slice. This is almost always the same as the length, but
    /// may be larger in case the allocator chose a larger size to align the range afterwards.
    pub fn capacity(&self) -> I {
        self.alloc_capacity
    }
    /// Check whether the slice is empty or not.
    pub fn is_empty(&self) -> bool {
        self.len() == I::zero()
    }
    /// Get the offset of the underlying possibly non-continuously-organized mmap, that was added
    /// as part of [`BufferPool::begin_expand`].
    pub fn mmap_offset(&self) -> I {
        self.mmap_start
    }
    /// Get the size of the mmap region this slice was allocated in.
    pub fn mmap_size(&self) -> I {
        self.mmap_size
    }
    /// Get the extra field from the mmap region this slice belongs to, copied.
    pub fn extra(&self) -> E {
        self.extra
    }
}
impl<'a, I, H, E, G> Drop for BufferSlice<'a, I, H, E, G>
where
    I: Integer,
    G: Guard,
    E: Copy,
    H: Handle<I, E>,
{
    fn drop(&mut self) {
        match self.reclaim_inner() {
            true => (),
            false => {
                log::debug!("Trying to drop a BufferSlice that is in use, leaking memory",);
            }
        }
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> ops::Deref
    for BufferSlice<'a, I, H, E, G>
{
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> ops::DerefMut
    for BufferSlice<'a, I, H, E, G>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> Borrow<[u8]>
    for BufferSlice<'a, I, H, E, G>
{
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> BorrowMut<[u8]>
    for BufferSlice<'a, I, H, E, G>
{
    fn borrow_mut(&mut self) -> &mut [u8] {
        self.as_slice_mut()
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> AsRef<[u8]>
    for BufferSlice<'a, I, H, E, G>
{
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl<'a, I: Integer, H: Handle<I, E>, E: Copy, G: Guard> AsMut<[u8]>
    for BufferSlice<'a, I, H, E, G>
{
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_slice_mut()
    }
}
/// A handle for expansion. When this handle is retrieved by the [`BufferPool::begin_expand`]
/// method, the range has already been reserved, so it's up to this handle to initialize it.
// TODO: Reclaim pending slice upon Drop or a fallible cancel method.
pub struct ExpandHandle<'a, I: Integer, H: Handle<I, E>, E: Copy> {
    offset: I,
    len: I,
    pool: &'a BufferPool<I, H, E>,
}
impl<'a, I, H, E> ExpandHandle<'a, I, H, E>
where
    I: Integer,
    H: Handle<I, E>,
    E: Copy,
{
    /// Get the length of the range that has been reserved for this specific allocation.
    pub fn len(&self) -> I {
        self.len
    }
    /// Check whether the pending expansion consists of no bytes. Note that zero-sized allocations
    /// will panic anyways, so this will always return false, but is still there for completeness.
    pub fn is_empty(&self) -> bool {
        self.len == I::zero()
    }
    /// Get the offset within the buffer pool, where this allocation is to take place.
    pub fn offset(&self) -> I {
        self.offset
    }
    /// Provide the handle with an actually allocated pointer, initializing the memory range within
    /// the buffer pool.
    ///
    /// # Safety
    ///
    /// For this to be safe, the pointer must be a valid allocation (anywhere) of the size
    /// originally inputted. The allocation must have the static lifetime, so pointers to the stack
    /// obviously don't apply here.
    pub unsafe fn initialize(self, pointer: NonNull<u8>, extra: E) {
        let mut mmap_write_guard = self.pool.mmap_map.write();

        let key = Offset(self.offset());

        let new_initialized_value = MmapInfo {
            addr: Addr::Initialized(pointer),
            size: Size::from_size(self.len()),
            extra: MaybeUninit::new(extra),
        };

        mmap_write_guard
            .insert(key, new_initialized_value)
            .expect("expected ExpandHandle to already have a pending mmap");

        // Before releasing the guard and allowing new slices from be acquired, we'll do a last
        // lock of the occ map, to mark the range as free.
        let mut occ_write_guard = self.pool.occ_map.write();
        let mut free_write_guard = self.pool.free_map.write();

        occ_write_guard
            .insert(
                OccOffset::from_offset_used(self.offset(), false),
                Size(self.len()),
            )
            .expect_none("expected newly-acquired slice not to conflict with any existing one");

        let updated =
            free_write_guard.insert(FreeEntry::from_size_offset(self.len(), self.offset()));
        assert!(updated);
    }
}
/// The strategy to use when allocating, with tradeoffs between heap fragmentation, and the
/// algorithmic complexity of allocating.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AllocationStrategy<I> {
    /// Allocate as optimally as possible, by taking some extra time into making sure that the
    /// smallest range with the smallest alignment possible is used. This is currently O(1) in best
    /// case, but O(n) in worst case.
    Optimal,

    /// Allocate as quickly as possible. This is currently implemented as a B-tree lookup in the
    /// free space map, so it'll prioritize smaller ranges, and smaller alignments afterwards. This
    /// is O(log n) in best case, average case, and worst case.
    Greedy,

    /// Allocate at a fixed offset. Since this won't cause the allocator to find a suitable range,
    /// but only check whether the range requested exists, this is also O(log n) in best case,
    /// average case, and worst case.
    Fixed(I),
}

impl<I> Default for AllocationStrategy<I> {
    fn default() -> Self {
        Self::Optimal
    }
}

type AcquireSliceRet<I, E> = (ops::Range<I>, I, ops::Range<I>, *mut u8, E);

/// The result originating from [`BufferPool::close`].
pub type CloseResult<I, H, E> =
    Result<(Option<H>, MmapEntries<I, E>), CloseError<BufferPool<I, H, E>>>;

impl<I, H, E> BufferPool<I, H, E>
where
    I: Integer,
    H: Handle<I, E>,
    E: Copy,
{
    /// Begin a buffer pool expansion, by reserving a new not-yet-usable "pending" mmap region.
    /// Internally, this will use an intent lock, so while there can be multiple parallel
    /// expansions in the actual allocation phase, the process of finding a new range within the
    /// pool for allocation, can only have one writer at a time. That said, readers of this pool
    /// will still be able to function correctly, and the only critical exclusive section, is only
    /// to insert the new range into the pool.
    pub fn begin_expand(
        &self,
        additional: I,
    ) -> Result<ExpandHandle<'_, I, H, E>, BeginExpandError> {
        let new_offset = {
            // Get an intent guard (in other words, upgradable read guard), which allows regular
            // readers to continue acquiring new slices etc, but only allows this thread to be able
            // to upgrade into an exclusive lock (which we do later during the actual insert).
            let mmap_intent_guard = self.mmap_map.upgradable_read();

            // Get the last mmapped range, no matter whether it's pending or ready to use.
            let greatest_offset = mmap_intent_guard.last_key_value().map_or(
                // If there somehow weren't any remaining mmap regions, we just implicitly set the
                // next offset to zero.
                Result::<I, BeginExpandError>::Ok(I::zero()),
                |(last_key, last_value)| {
                    let start = last_key
                        .0
                        .checked_add(last_value.size.0)
                        .ok_or(BeginExpandError)?;
                    let _end = start.checked_add(additional).ok_or(BeginExpandError)?;
                    Ok(start)
                },
            );

            let new_offset = match greatest_offset {
                Ok(o) => o,
                Err(_) => {
                    // If we are using small indices, and we run out of new ranges, we have no
                    // choice but to do an O(n) search to find a new free range. Remember though,
                    // that these are the mmaps, which are not supposed to be allocated and freed
                    // that often, so this case is relatively uncommon.
                    mmap_intent_guard
                        .iter()
                        .find_map(|(k, v)| {
                            let start = k.0.checked_add(v.size.0)?;
                            let _end = start.checked_add(additional)?;
                            Some(start)
                        })
                        .ok_or(BeginExpandError)?
                }
            };

            let mut mmap_write_guard = RwLockUpgradableReadGuard::upgrade(mmap_intent_guard);

            // Insert a new region marked as "pending", with an uninitialized pointer.
            let new_info = MmapInfo::null();
            let prev = mmap_write_guard.insert(Offset(new_offset), new_info);

            assert!(prev.is_none());

            // Implicitly drop the intent guard, allowing other threads to also expand this buffer
            // pool. There is no race condition here whatsoever, since we have marked our
            // prereserved range as "pending".
            new_offset
        };
        Ok(ExpandHandle {
            offset: new_offset,
            len: additional,
            pool: self,
        })
    }

    // TODO: Shrink support

    /// Retrieve the user "handle" that was passed to this buffer pool at initialization.
    pub fn handle(&self) -> Option<&H> {
        self.handle.as_ref()
    }

    /// Attempt to close, and thus free each buffer slice owned by the entire pool, returning the
    /// handle and the mmap ranges if present.
    ///
    /// If there are any pending commands that have guarded buffer slices from this pool, the
    /// entire memory will be leaked, for now.
    pub fn try_close(mut self) -> CloseResult<I, H, E> {
        if self.guarded_occ_count.load(Ordering::Acquire) > 0 {
            // TODO: Free as much memory as possible when this happens, rather than the entire
            // pool. This may add unnecessary overhead though, if a separate guard count would be
            // stored for every mmap info entry.
            return Err(CloseError { this: self });
        }

        let handle = self.handle.take();
        let mmap_map = mem::replace(self.mmap_map.get_mut(), BTreeMap::new());
        let entries = MmapEntries {
            inner: mmap_map.into_iter(),
        };

        Ok((handle, entries))
    }
    /// Create a new empty buffer pool, using an optional user "handle" that is stored together
    /// with the rest of the pool.
    pub fn new(handle: Option<H>) -> Self {
        Self {
            occ_map: RwLock::new(BTreeMap::new()),
            mmap_map: RwLock::new(BTreeMap::new()),
            free_map: RwLock::new(BTreeSet::new()),
            guarded_occ_count: AtomicUsize::new(0),
            handle,
            options: BufferPoolOptions::default(),
        }
    }
    /// Set the allocation options used by the buffer pool, e.g. the size and alignment bounds.
    pub fn with_options(mut self, options: BufferPoolOptions<I>) -> Self {
        self.options = options;
        self
    }
    // Tries to acquire a buffer slice by inserting an occupied entry into the occ map. The buffer
    // slice must not be able to span multiple mmaps, since their base pointers may not be
    // continuous.  Returns the range of new newly occupied entry, the range of that entry's mmap,
    // the base pointer of that entry's mmap, and the extra data associated with the mmap.
    fn acquire_slice(
        &self,
        len: I,
        alignment: I,
        strategy: AllocationStrategy<I>,
    ) -> Option<AcquireSliceRet<I, E>> {
        assert_ne!(len, I::zero());

        if len > self.options.maximum_size {
            return None;
        }

        if alignment > (I::from(1u8) << self.options.log2_maximum_alignment) {
            return None;
        }
        let alignment = cmp::max(
            alignment,
            I::from(1u8) << self.options.log2_minimum_alignment,
        );

        // Begin by obtaining an intent guard. This will unfortunately prevent other threads from
        // simultaneously searching the map for partitioning it; however, there can still be other
        // threads checking whether it's safe to munmap certain offsets.
        let occ_intent_guard = self.occ_map.upgradable_read();
        let free_intent_guard = self.free_map.upgradable_read();

        fn align<I: Integer>(off: I, alignment: I) -> Option<I> {
            assert_ne!(alignment, I::from(0u8));
            assert!(alignment.is_power_of_two());

            if alignment == I::from(1u8) {
                return Some(off);
            }

            off.checked_add(alignment - I::from(1u8))?
                .checked_div(alignment)?
                .checked_mul(alignment)
        }

        let (occ_k, occ_v, free_e) = if let AllocationStrategy::Fixed(at) = strategy {
            let (occ_k, occ_v) = occ_intent_guard
                .range(..=OccOffset::from_offset_used(at, false))
                .next_back()?;

            if occ_v.size() < len {
                return None;
            }

            let free_e = free_intent_guard
                .get(&FreeEntry::from_size_offset(occ_v.0, occ_k.offset()))
                .expect("expected occ map to contain a corresponding entry for the free entry");

            (occ_k, occ_v, *free_e)
        } else {
            fn find_o_n<I: Integer>(
                free_map: &BTreeSet<FreeEntry<I>>,
                len: I,
                alignment: I,
            ) -> Option<&FreeEntry<I>> {
                // This is the O(n) allocation mechanism, that always finds a suitable range to
                // use, unless the pool is full.
                //
                // The O(n) allocation algorithm is used when the "Optimal" allocation strategy has
                // been specified.
                free_map.iter().find(|e| {
                    e.size() >= len
                        && align(e.offset(), alignment)
                            .map_or(false, |aligned| e.size() - (aligned - e.offset()) >= len)
                })
            }
            fn find_o_logn<I: Integer>(
                free_map: &BTreeSet<FreeEntry<I>>,
                len: I,
                alignment: I,
            ) -> Option<&FreeEntry<I>> {
                // This is the O(log n) allocation mechanism, that works in _most_ cases.
                //
                // Get a new entry to pop from the free space B-tree. The key Ord impl first
                // compares the size, and then the alignment (or rather log2 of the alignment, but
                // log2 n > log2 m implies n > m for natural numbers n,m > 1), in that order. By
                // doing a half-open range with the start, bound, this will prioritize the smallest
                // size and alignments, trying to reduce pool fragmentation.
                //
                // The alignment is directly calculated based on the number of leading binary zeroes of the
                // offset (which is the value); in other words, it'll be the largest power of two that
                // divides the offset.
                //
                // Because of that, smaller sizes will be preferred over larger sizes, but aligned
                // ranges may however, be fewer.
                //
                // Note that this isn't perfect. Size+alignment doesn't work very well with B-trees
                // (maybe a quadtree or similar for that?). Note that not adequately aligned ranges
                // that are still large enough to account for misalignment, will still work. At the
                // moment, the solution is to naïvely do another O(n) search, finding a suitable
                // range. TODO: Fix that.
                let item = free_map
                    .range(FreeEntry::from_size_offset(len, alignment)..)
                    .next()?;

                let check_if_misalignment_would_work =
                    |aligned| item.size() - (aligned - item.offset()) >= len;

                if item.size() >= len
                    && align(item.offset(), alignment)
                        .map_or(false, check_if_misalignment_would_work)
                {
                    Some(item)
                } else {
                    find_o_n(free_map, len, alignment)
                }
            }
            let free_e = if let AllocationStrategy::Greedy = strategy {
                find_o_logn(&*free_intent_guard, len, alignment)?
            } else if let AllocationStrategy::Optimal = strategy {
                find_o_n(&*free_intent_guard, len, alignment)?
            } else {
                unreachable!()
            };
            let (occ_k, occ_v) = occ_intent_guard
                .get_key_value(&OccOffset::from_offset_used(free_e.offset(), false))
                .expect("expected free map to contain a corresponding entry for the occ entry");

            (occ_k, occ_v, *free_e)
        };

        assert!(!occ_k.is_used());
        assert_eq!(occ_k.offset(), free_e.offset());
        assert_eq!(
            I::from(u8::try_from(occ_k.offset().trailing_zeros()).unwrap()),
            free_e.log2_of_alignment()
        );
        assert_eq!(occ_v.0, free_e.size());

        let original_off = free_e.offset();

        let aligned_off =
            align(original_off, alignment).expect("bypassed alignment check in iterator");
        assert!(aligned_off < free_e.offset() + free_e.size());
        let align_advancement = aligned_off - free_e.offset();

        let new_offset = {
            let mut occ_write_guard = RwLockUpgradableReadGuard::upgrade(occ_intent_guard);
            let mut free_write_guard = RwLockUpgradableReadGuard::upgrade(free_intent_guard);

            /*let mut occ_v = occ_write_guard
            .remove(occ_k)
            .expect("expected entry not to be removed by itself when acquiring slice");*/

            let had_prev = free_write_guard.remove(&free_e);
            assert!(had_prev);

            let prev = occ_write_guard.remove(&OccOffset::from_offset_used(original_off, false));
            assert!(prev.is_some());

            if free_e.size() > len {
                let mut upper_free_e = free_e;
                // Reinsert the upper part of the free range, if the entire range wasn't used.
                upper_free_e.set_size(upper_free_e.size() - len);
                upper_free_e.set_offset(upper_free_e.offset() + len);

                let updated = free_write_guard.insert(upper_free_e);
                assert!(updated);

                let prev = occ_write_guard.insert(
                    OccOffset::from_offset_used(upper_free_e.offset(), false),
                    Size(upper_free_e.size()),
                );
                assert_eq!(prev, None);
            }
            if align_advancement > I::zero() {
                // If there was unused space due to alignment, insert that small region marked
                // unused as well.
                let new_free_e = FreeEntry::from_size_offset(align_advancement, original_off);

                let updated = free_write_guard.insert(new_free_e);
                assert!(
                    updated,
                    "somehow the small alignment region was already mapped"
                );

                let prev = occ_write_guard.insert(
                    OccOffset::from_offset_used(original_off, false),
                    Size(new_free_e.size()),
                );
                assert!(prev.is_none());
            }

            let new_offset = aligned_off;
            let new_occ_k = OccOffset::from_offset_used(new_offset, true);
            let new_occ_v = Size(len);
            occ_write_guard
                .insert(new_occ_k, new_occ_v)
                .expect_none("expected new entry not to already be inserted");

            new_offset
        };
        let (mmap_range, pointer, extra) = {
            let mmap_read_guard = self.mmap_map.read();

            let (mmap_k, mmap_v) = mmap_read_guard
                .range(..=Offset(new_offset))
                .next_back()
                .expect(
                    "expected all free entries in the occ map to have a corresponding mmap entry",
                );

            let mmap_start = mmap_k.0;
            let mmap_size = mmap_v.size;
            let mmap_end = mmap_start
                .checked_add(mmap_size.0)
                .expect("expected mmap end not to overflow u32::MAX");

            assert!(mmap_v.is_init());
            assert!(mmap_start <= new_offset);
            assert!(mmap_end >= new_offset + len);

            let (extra, pointer) = unsafe {
                assert_ne!(
                    mmap_v.size.0,
                    I::zero(),
                    "expected found slice to not have size zero"
                );

                // SAFETY: The following assumption is safe, because we have already checked that
                // the pointer is initialized, which implies that the extra field also has to be.
                let extra = mmap_v.extra.assume_init();

                // SAFETY: Here, we're still manipulating a raw pointer, so there wouldn't really
                // be unsoundness apart from a possible overflow, but it's also otherwise safe
                // because we have asserted that the length is nonzero.
                let base_pointer = mmap_v.addr;

                let pointer = base_pointer
                    .as_ptr()
                    .add((new_offset - mmap_k.0).try_into_usize().unwrap())
                    as *mut u8;

                (extra, pointer)
            };
            (mmap_start..mmap_end, pointer, extra)
        };

        let offset = aligned_off;

        // TODO
        let actual_len = len;

        Some((offset..offset + actual_len, len, mmap_range, pointer, extra))
    }
    fn construct_buffer_slice<G: Guard>(
        alloc_range: ops::Range<I>,
        alloc_len: I,
        mmap_range: ops::Range<I>,
        pointer: *mut u8,
        extra: E,
        pool: PoolRefKind<I, H, E>,
    ) -> BufferSlice<'_, I, H, E, G> {
        debug_assert!(alloc_len <= alloc_range.end - alloc_range.start);

        BufferSlice {
            alloc_start: alloc_range.start,
            alloc_capacity: alloc_range.end - alloc_range.start,
            alloc_len,

            mmap_start: mmap_range.start,
            mmap_size: mmap_range.end - mmap_range.start,
            pointer,
            pool,
            extra,
            guard: None,
        }
    }
    /// Try to acquire a statically (as in compiler-checked and lifetime-tied) borrowed slice, from
    /// this buffer. The slice will automatically be reclaimed upon drop, so long as there is no
    /// guard protecting the slice at that time. If there is, the memory will be leaked instead,
    /// and the pool will not be able to use the offset, as it will be marked "occpied" and nothing
    /// will free it.
    pub fn acquire_borrowed_slice<G: Guard>(
        &self,
        len: I,
        alignment: I,
        strategy: AllocationStrategy<I>,
    ) -> Option<BufferSlice<'_, I, H, E, G>> {
        let (alloc_range, alloc_len, mmap_range, pointer, extra) =
            self.acquire_slice(len, alignment, strategy)?;
        Some(Self::construct_buffer_slice(
            alloc_range,
            alloc_len,
            mmap_range,
            pointer,
            extra,
            PoolRefKind::Ref(self),
        ))
    }
    /// Try to acquire a weakly-borrowed ([`std::sync::Weak`]) slice, that may outlive this buffer
    /// pool. If that would happen, most functionality of the slice would cause it to panic,
    /// although this can be checked for as well.
    ///
    /// These slices can also be guarded, see [`acquire_borrowed_slice`] for a detailed explanation
    /// of that.
    ///
    /// [`acquire_borrowed_slice`]: #method.acquire_borrowed_slice
    pub fn acquire_weak_slice<G: Guard>(
        self: &Arc<Self>,
        len: I,
        alignment: I,
        strategy: AllocationStrategy<I>,
    ) -> Option<BufferSlice<'static, I, H, E, G>> {
        let (alloc_range, alloc_len, mmap_range, pointer, extra) =
            self.acquire_slice(len, alignment, strategy)?;
        Some(Self::construct_buffer_slice(
            alloc_range,
            alloc_len,
            mmap_range,
            pointer,
            extra,
            PoolRefKind::Weak(Arc::downgrade(self)),
        ))
    }
    /// Try to acquire a strongly-borrowed ([`std::sync::Arc`]) slice, that ensures this buffer
    /// pool cannot be outlived by preventing the whole pool from being dropped.
    ///
    /// These slices can also be guarded, see [`acquire_borrowed_slice`] for a detailed explanation
    /// of that.
    ///
    /// [`acquire_borrowed_slice`]: #method.acquire_borrowed_slice
    pub fn acquire_strong_slice<G: Guard>(
        self: &Arc<Self>,
        len: I,
        alignment: I,
        strategy: AllocationStrategy<I>,
    ) -> Option<BufferSlice<'static, I, H, E, G>> {
        let (alloc_range, alloc_len, mmap_range, pointer, extra) =
            self.acquire_slice(len, alignment, strategy)?;
        Some(Self::construct_buffer_slice(
            alloc_range,
            alloc_len,
            mmap_range,
            pointer,
            extra,
            PoolRefKind::Strong(Arc::clone(self)),
        ))
    }
    fn remove_free_offset_below(
        free_map: &mut BTreeSet<FreeEntry<I>>,
        occ_map: &mut BTreeMap<OccOffset<I>, Size<I>>,
        mmap_map: &BTreeMap<Offset<I>, MmapInfo<I, E>>,
        start: &mut I,
        size: &mut I,
    ) -> bool {
        let previous_start = *start;
        let lower_offset = match previous_start.checked_sub(I::from(1u8)) {
            Some(l) => l,
            None => return false,
        };

        let (cur_mmap_k, cur_mmap_v) = mmap_map
            .range(..=Offset(previous_start))
            .next_back()
            .unwrap();

        assert!(cur_mmap_k.0 + cur_mmap_v.size.0 > previous_start);
        assert!(cur_mmap_k.0 <= previous_start);

        // We can't merge free entries faster than O(n), because the obviously have to be coninuous
        // for that to work, and they are only laid out based on size and alignment. What we can do
        // though, is to merge the occ map entries if possible, and then indirectly merging the
        // free entries.

        let lower_occ_partial_k = OccOffset::from_offset_used(lower_offset, false);

        if let Some((lower_occ_k, lower_occ_v)) = occ_map.range(..=lower_occ_partial_k).next_back()
        {
            let lower_occ_k = *lower_occ_k;
            let lower_occ_v = *lower_occ_v;

            assert!(!lower_occ_k.is_used());

            if lower_occ_k.offset() + lower_occ_v.0 != previous_start {
                // There is another occupied range between these.
                return false;
            }

            let (mmap_k, _) = mmap_map
                .range(..=Offset(lower_occ_k.offset()))
                .next_back()
                .unwrap();

            if mmap_k != cur_mmap_k {
                // The range cannot be merged as its underlying memory range is not continuous.
                return false;
            }

            let lower_occ_v_again = occ_map
                .remove(&lower_occ_k)
                .expect("expected previously found key to exist in the b-tree map");

            assert_eq!(lower_occ_v_again, lower_occ_v);

            let had_prev = free_map.remove(&FreeEntry::from_size_offset(
                lower_occ_v.size(),
                lower_occ_k.offset(),
            ));

            assert!(had_prev);

            *start = lower_occ_k.offset();
            *size += lower_occ_v.size();

            true
        } else {
            false
        }
    }
    fn remove_free_offset_above(
        free_map: &mut BTreeSet<FreeEntry<I>>,
        occ_map: &mut BTreeMap<OccOffset<I>, Size<I>>,
        mmap_map: &BTreeMap<Offset<I>, MmapInfo<I, E>>,
        start: &mut I,
        size: &mut I,
    ) -> bool {
        if *size == I::from(0u8) {
            return false;
        }

        let end = *start + *size;

        let higher_occ_k = OccOffset::from_offset_used(end, false);

        let (cur_mmap_k, cur_mmap_v) = mmap_map.range(..=Offset(*start)).next_back().unwrap();

        assert!(cur_mmap_k.0 + cur_mmap_v.size.0 > *start);
        assert!(cur_mmap_k.0 <= *start);

        if cur_mmap_k.0 + cur_mmap_v.size.0 == *start {
            // The mmap range ended at the current offset, which makes it impossible to merge the
            // above offset into a single region, since all occupiable ranges must be continuous.
            return false;
        }

        if let Some(higher_occ_v) = occ_map.remove(&higher_occ_k) {
            let had_prev = free_map.remove(&FreeEntry::from_size_offset(
                higher_occ_v.size(),
                higher_occ_k.offset(),
            ));

            assert!(had_prev);

            *size += higher_occ_v.size();

            true
        } else {
            false
        }
    }

    unsafe fn reclaim_slice_inner<G: Guard>(&self, slice: &BufferSlice<'_, I, H, E, G>) {
        let mut occ_write_guard = self.occ_map.write();
        let mut free_write_guard = self.free_map.write();

        let mut start = slice.alloc_start;
        let mut size = slice.alloc_capacity;

        let occ_v = occ_write_guard
            .remove(&OccOffset::from_offset_used(start, true))
            .expect("expected occ map to contain buffer slice when reclaiming it");

        let mmap_guard = self.mmap_map.read();

        assert_eq!(occ_v.size(), slice.alloc_capacity);

        while Self::remove_free_offset_below(
            &mut *free_write_guard,
            &mut *occ_write_guard,
            &*mmap_guard,
            &mut start,
            &mut size,
        ) {}
        while Self::remove_free_offset_above(
            &mut *free_write_guard,
            &mut *occ_write_guard,
            &*mmap_guard,
            &mut start,
            &mut size,
        ) {}

        let new_free_e = FreeEntry::from_size_offset(size, start);

        let updated = free_write_guard.insert(new_free_e);

        assert!(
            updated,
            "expected newly resized free range not to start existing again before insertion",
        );

        let new_occ_k = OccOffset::from_offset_used(start, false);
        let new_occ_v = Size::from_size(size);

        occ_write_guard.insert(new_occ_k, new_occ_v).unwrap_none();
    }
    fn drop_impl(&mut self) {
        let count = self.guarded_occ_count.load(Ordering::Acquire);

        if count == 0 {
            if let Some(h) = self.handle.take() {
                // This won't allocate, since the new mmap is entry.
                let entries = mem::replace(self.mmap_map.get_mut(), BTreeMap::new());
                let _ = h.close_all(MmapEntries {
                    inner: entries.into_iter(),
                });
            }
        } else {
            log::warn!("Leaking parts of the buffer pool, since there were {} slices that were guarded by futures that haven't been completed", count);
        }
    }
    /// Returns the number of active guards that are used in the pool.
    ///
    /// This method is O(1) and doesn't count anything; it simply fetches an internal counter.
    pub fn active_guard_count(&self) -> usize {
        self.guarded_occ_count.load(Ordering::Relaxed)
    }
}

impl<I: Integer, H: Handle<I, E>, E: Copy> Drop for BufferPool<I, H, E> {
    fn drop(&mut self) {
        self.drop_impl();
    }
}

/// The iterator given to the close handle, or optionally retrieved by manually destroying a buffer
/// pool, that contains all the underlying allocations that the pool has been expanded with.
///
/// The iterator yields [`MmapEntry`].
#[derive(Debug)]
pub struct MmapEntries<I, E: Copy> {
    inner: ::alloc::collections::btree_map::IntoIter<Offset<I>, MmapInfo<I, E>>,
}
/// The entry type from the [`MmapEntries`] iterator, that contains all the information that the
/// buffer pool had about that mmap, when it's being destroyed.
///
/// Only handles that were actually initialized here, are listed. A reserved range that was still
/// pending when this iterator was created, that had its expansion handle dropped, will not be
/// included in the iterator.
#[derive(Debug)]
pub struct MmapEntry<I, E> {
    /// The internally allocated offset within the buffer pool, where the mmap was put.
    pub pool_offset: I,

    /// The size that the mmap allocation had. This is exactly the same as the size inputted as
    /// part of [`BufferPool::begin_expand`].
    pub size: I,

    /// The pointer that was given as part of [`ExpandHandle::initialize`].
    ///
    /// This pointer is _guaranteed_ to be a valid allocation of size [`size`], so long as the
    /// allocation given to the pool as part of [`ExpandHandle::initialize`] was valid (which would
    /// immediately be undefined behavior in the first case).
    ///
    /// [`size`]: #structfield.size
    pub pointer: NonNull<u8>,

    /// The extra field that was also supplied to [`ExpandHandle::initialize`].
    pub extra: E,
}
impl<I, E> Iterator for MmapEntries<I, E>
where
    E: Copy,
{
    type Item = MmapEntry<I, E>;

    fn next(&mut self) -> Option<Self::Item> {
        'entries: loop {
            let (offset, info) = self.inner.next()?;

            let pointer = match info.addr {
                Addr::Initialized(pointer) => pointer,
                Addr::Uninitialized => continue 'entries,
            };

            return Some(MmapEntry {
                pool_offset: offset.0,
                size: info.size.0,
                pointer,

                // SAFETY: The following unsafe block is safe, because the MmapInfo struct is
                // organized in a way that the address enum indicates whether nor not the struct
                // has been initialized. Since the earlier match statement has already checked that
                // the pointer be initialized, and thus the struct, we can assume_init here.
                extra: unsafe { info.extra.assume_init() },
            });
        }
    }
}

/// The requirement of a handle to be able to be passed into the buffer pool.
///
/// This trait is only currently used in the destructor of the buffer pool, after all the ranges
/// have been validated not to be in use by an active guard.
pub trait Handle<I, E: Copy>: Sized {
    /// The possible error that may occur when freeing one or more mmap entries. This error type is
    /// forwarded to the buffer pool when this handle is used, which only currently happens in the
    /// destructor of [`BufferPool`].
    type Error;

    /// The function called when a buffer pool wants one or more ranges, proven not to contain
    /// guarded slices, to be deallocated.
    ///
    /// This function is only called directly when a full close failed due to active guards, or in
    /// the default implementation of [`close_all`].
    ///
    /// [`close_all`]: #method.close_all
    fn close(&mut self, mmap_entries: MmapEntries<I, E>) -> Result<(), Self::Error>;

    /// The function called when a buffer pool is dropped.
    ///
    /// All the mmap ranges (originating from [`begin_expand`]) that have been initialized, are
    /// also included here. The reason this function exists, is to allow for more performant memory
    /// deallocation, when it is known that the buffer pool did not contain any guarded buffer slice.
    ///
    /// An implementor might for example, close the file descriptor rather than repeatedly calling
    /// _munmap(2)_, if _mmap(2)_ is used internally.
    ///
    /// [`begin_expand`]: ./struct.BufferPool.html#method.begin_expand
    fn close_all(mut self, mmap_entries: MmapEntries<I, E>) -> Result<(), Self::Error> {
        self.close(mmap_entries)
    }
}
/// The requirement of a guard to a slice. Guards can optionally prevent slices from being
/// reclaimed; the slices have a fallible [`BufferSlice::reclaim`] method, but their `Drop` impl
/// will cause the memory to leak if it's still protected by the guard. This is especially useful
/// when the buffers are shared with another process (`io_uring` for instance), or by hardware,
/// since this prevents data races that could occur, if a new buffer for a different purpose
/// happens to use the same memory as an old buffer that e.g. hardware thinks it can write to.
pub trait Guard {
    /// Try to release the guard, returning either true for success or false for failure.
    fn try_release(&self) -> bool;
}

/// The potential error from [`BufferSlice::with_guard`] or [`BufferSlice::guard`], indicating that
/// a different guard is already in use by the the buffer slice. Since that method takes self by
/// value, the old self is included here, to allow for reuse in case of failure.
///
/// This corresponds to `EEXIST` if the `redox` feature is enabled.
pub struct WithGuardError<T> {
    /// The self passed by value that could have it's guard type replaced.
    pub this: T,
}
impl<T> fmt::Debug for WithGuardError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WithGuardError")
            // TODO
            .finish()
    }
}

impl<T> fmt::Display for WithGuardError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to replace guard, due to another guard already existing"
        )
    }
}
#[cfg(any(test, feature = "std"))]
impl<T> std::error::Error for WithGuardError<T> {}

/// The potential error from [`BufferSlice::reclaim`], caused by the slice being guarded.
///
/// This error is convertible to `EADDRINUSE`, if the `redox` feature is enabled.
pub struct ReclaimError<T> {
    /// The slice that couldn't be reclaimed, due to an active guard.
    pub this: T,
}
impl<T> fmt::Display for ReclaimError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to reclaim buffer slice, since it was in use")
    }
}
impl<T> fmt::Debug for ReclaimError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReclaimError").finish()
    }
}

#[cfg(any(test, feature = "std"))]
impl<T> std::error::Error for ReclaimError<T> {}

/// The error from [`BufferPool::close`], meaning that there is currently a guarded buffer slice in
/// use by the pool, preventing resource freeing.
///
/// As with [`ReclaimError`], this error is convertible to `EADDRINUSE` in case the `redox` feature
/// is enabled.
#[derive(Debug)]
pub struct CloseError<T> {
    /// The buffer pool that couldn't be destroyed.
    pub this: T,
}
impl<T> fmt::Display for CloseError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to close since buffers were in use")
    }
}
/// The error internally caused by arithmetic overflow, that indicates the buffer pool has no more
/// usable ranges.
///
/// This error can be converted into `ENOMEM` if the `redox` feature is enabled.
#[derive(Debug)]
pub struct BeginExpandError;

impl fmt::Display for BeginExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to expand buffer: no more buffer pool memory (arithmetic overflow)"
        )
    }
}

#[cfg(any(test, feature = "std"))]
impl std::error::Error for BeginExpandError {}

#[cfg(feature = "redox")]
mod libc_error_impls {
    use super::*;

    use syscall::error::Error;
    use syscall::error::{EADDRINUSE, EEXIST, ENOMEM};

    impl From<BeginExpandError> for Error {
        fn from(_: BeginExpandError) -> Error {
            Error::new(ENOMEM)
        }
    }
    impl<T> From<CloseError<T>> for Error {
        fn from(_: CloseError<T>) -> Error {
            Error::new(EADDRINUSE)
        }
    }
    impl<T> From<ReclaimError<T>> for Error {
        fn from(_: ReclaimError<T>) -> Error {
            Error::new(EADDRINUSE)
        }
    }
    impl<T> From<WithGuardError<T>> for Error {
        fn from(_: WithGuardError<T>) -> Error {
            Error::new(EEXIST)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{mem, thread};

    fn setup_pool(
        maps: impl IntoIterator<Item = Vec<u8>>,
        options: BufferPoolOptions<u32>,
    ) -> (BufferPool<u32, NoHandle, ()>, u32) {
        let pool = BufferPool::new(None);
        let mut total_size = 0;

        for map in maps {
            let mut slice = map.into_boxed_slice();

            let ptr = slice.as_mut_ptr();
            let len = slice.len();

            total_size += u32::try_from(len).unwrap();

            let _raw_slice = Box::into_raw(slice);
            unsafe {
                pool.begin_expand(u32::try_from(len).unwrap())
                    .unwrap()
                    .initialize(std::ptr::NonNull::new(ptr).unwrap(), ())
            }
        }
        (pool.with_options(options), total_size)
    }
    fn setup_default_pool(options: BufferPoolOptions<u32>) -> (BufferPool<u32, NoHandle, ()>, u32) {
        setup_pool(
            vec![vec![0u8; 32768], vec![0u8; 4096], vec![0u8; 65536]],
            options,
        )
    }

    #[test]
    fn occ_map_acquisition_single_mmap_optimal() {
        occ_map_acquisition_single_mmap(AllocationStrategy::Optimal)
    }
    #[test]
    fn occ_map_acquisition_single_mmap_greedy() {
        occ_map_acquisition_single_mmap(AllocationStrategy::Greedy)
    }

    fn occ_map_acquisition_single_mmap(strategy: AllocationStrategy<u32>) {
        let (pool, _) = setup_default_pool(Default::default());

        let mut slices = Vec::new();

        loop {
            let mut slice = match pool.acquire_borrowed_slice::<NoGuard>(4096, 1, strategy) {
                Some(s) => s,
                None => break,
            };

            let text = b"Hello, world!";
            slice[..text.len()].copy_from_slice(text);
            assert_eq!(&slice[..text.len()], text);
            slices.push(slice);
        }
        drop(slices);

        mem::forget(pool);
    }
    #[test]
    fn occ_multithreaded_optimal() {
        occ_multithreaded(AllocationStrategy::Optimal)
    }
    #[test]
    fn occ_multithreaded_greedy() {
        occ_multithreaded(AllocationStrategy::Optimal)
    }

    fn occ_multithreaded(strategy: AllocationStrategy<u32>) {
        // This test is not about aliasing, but rather to get all the assertions and expects, to
        // work when there are multiple threads constantly trying to acquire and release slices.

        let (pool, _) = setup_default_pool(Default::default());
        let pool = pool.shared();

        const THREAD_COUNT: usize = 8;

        #[cfg(not(miri))]
        const N: usize = 1000;

        #[cfg(miri)]
        const N: usize = 128;

        let threads = (0..THREAD_COUNT).map(|_| {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                use rand::Rng;

                let mut thread_rng = rand::thread_rng();

                for _ in 0..N {
                    'retry: loop {
                        let len = thread_rng.gen_range(64, 4096);
                        let align = 1 << thread_rng.gen_range(0, 3);
                        match pool.acquire_borrowed_slice::<NoGuard>(len, align, strategy) {
                            Some(_) => break 'retry,
                            None => continue 'retry,
                        }
                    }
                }
            })
        });
        for thread in threads {
            thread.join().unwrap();
        }
    }

    #[test]
    fn no_aliasing_optimal() {
        no_aliasing(AllocationStrategy::Optimal)
    }
    #[test]
    fn no_aliasing_greedy() {
        no_aliasing(AllocationStrategy::Greedy)
    }

    fn no_aliasing(strategy: AllocationStrategy<u32>) {
        let (pool, _) = setup_default_pool(Default::default());
        const SIZE: u32 = 512;

        let mut slices = Vec::new();

        loop {
            let slice = match pool.acquire_borrowed_slice::<NoGuard>(SIZE, 1, strategy) {
                Some(s) => s,
                None => break,
            };
            slices.push(slice);
        }
        for slice in &mut slices {
            assert!(slice.iter().all(|&byte| byte == 0));
            slice.fill(63);
        }
    }
    #[test]
    fn alignment_greedy() {
        alignment(AllocationStrategy::Greedy)
    }

    #[test]
    fn alignment_optimal() {
        alignment(AllocationStrategy::Optimal)
    }

    fn alignment(strategy: AllocationStrategy<u32>) {
        let options = BufferPoolOptions::default().with_minimum_alignment(1);
        let (pool, _) = setup_pool(vec![vec![0u8; 4096]], options);

        dbg!(options);

        fn get_and_check_slice(
            pool: &BufferPool<u32, NoHandle, ()>,
            size: u32,
            align: u32,
            fill_byte: u8,
            strategy: AllocationStrategy<u32>,
        ) -> BufferSlice<u32, NoHandle, ()> {
            let mut slice = pool.acquire_borrowed_slice(size, align, strategy).unwrap();
            assert!(slice.iter().all(|&byte| byte == 0));
            slice.fill(fill_byte);
            assert!(slice.iter().all(|&byte| byte == fill_byte));
            assert_eq!(slice.len(), size);
            assert_eq!(slice.offset() % align, 0);
            slice
        }

        {
            let _small_begin_slice = get_and_check_slice(&pool, 64, 1, 0x01, strategy);
            let _aligned_slice = get_and_check_slice(&pool, 128, 128, 0x02, strategy);
            let _half_page = get_and_check_slice(&pool, 2048, 2048, 0xFE, strategy);
        }
    }
    #[test]
    fn free_entry() {
        let mut entry = FreeEntry::from_size_offset(1024u32, 64);
        assert_eq!(entry.size(), 1024);
        assert_eq!(entry.offset(), 64);
        assert_eq!(entry.log2_of_alignment(), 6);

        entry.set_offset(128);
        assert_eq!(entry.size(), 1024);
        assert_eq!(entry.offset(), 128);
        assert_eq!(entry.log2_of_alignment(), 7);

        entry.set_offset(3);
        entry.set_size(4);
        assert_eq!(entry.size(), 4);
        assert_eq!(entry.offset(), 3);
        assert_eq!(entry.log2_of_alignment(), 0);
    }
}
