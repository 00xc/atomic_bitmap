#![cfg_attr(not(test), no_std)]

//! An atomic bitmap implementation for concurrent systems.
//!
//! This crate performs no heap allocations and uses `#[no_std]`.
//!
//! The core functionality is implementeted in the [`AtomicBitmap`]
//! trait, The main type that implements this trait is
//! [`FixedBitmap`], an implementation using fixed-size arrays.
//! However, the trait is implemented for any type that can be
//! represented as a slice of `AtomicU64`'s, like vectors.
//!
//! # Example
//!
//! ```rust
//! use atomic_bitmap::{AtomicBitmap, FixedBitmap};
//! use core::sync::atomic::Ordering::*;
//!
//! // A bitmap with 128 bits, all set to 0.
//! let map = FixedBitmap::<2>::new(false);
//!
//! // Setting a bit and getting its previous value
//! assert_eq!(map.set(4, true, Relaxed), Some(false));
//!
//! // Setting an invalid bit
//! assert_eq!(map.set(128, true, Relaxed), None);
//!
//! // Conditionally setting a bit
//! assert_eq!(
//! 	map.compare_exchange(
//! 		102, false, true, Relaxed, Relaxed, Relaxed
//! 	),
//! 	Some(Ok(false)),
//! );
//!
//! // Getting the value of a bit
//! assert_eq!(map.get(4, Relaxed), Some(true));
//!
//! // Getting the value of an invalid bit
//! assert_eq!(map.get(128, Relaxed), None);
//!
//! // Clearing the lowest set bit
//! assert_eq!(map.clear_lowest_one(Relaxed, Relaxed), Some(4));
//!
//! // Setting the lowest unset bit
//! assert_eq!(map.set_lowest_zero(Relaxed, Relaxed), Some(0));
//! ```

use core::array;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicU64, Ordering};

const SLOT_BITS: usize = u64::BITS as usize;
const SLOT_MASK: usize = SLOT_BITS - 1;

/// A trait implemented by all types that can act as an atomic bitmap.
pub trait AtomicBitmap {
	/// A slice of the inner representation.
	fn slots(&self) -> &[AtomicU64];

	/// Get the bit at the specified index given a memory ordering, or
	/// [`None`] if the index is out of bounds.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Within bounds
	/// assert_eq!(map.get(64, Relaxed), Some(false));
	///
	/// // Out of bounds
	/// assert_eq!(map.get(128, Relaxed), None);
	/// ```
	fn get(&self, idx: usize, order: Ordering) -> Option<bool> {
		let slot = self.slots().get(idx / SLOT_BITS)?.load(order);
		let mask = 1 << (idx & SLOT_MASK);
		Some(slot & mask != 0)
	}

	/// Identical to [`AtomicBitmap::get()`] but without any bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must guarantee that the index is within bounds of
	/// the bitmap.
	unsafe fn get_unchecked(
		&self,
		idx: usize,
		order: Ordering,
	) -> bool {
		let slot =
			self.slots().get_unchecked(idx / SLOT_BITS).load(order);
		let mask = 1 << (idx & SLOT_MASK);
		slot & mask != 0
	}

	/// Sets the specified bit to the given value and return the
	/// previous value, or [`None`] if the index is out of bounds.
	///
	/// The memory ordering corresponds to that of
	/// [`fetch_or()`](AtomicU64::fetch_or) and
	/// [`fetch_and()`]((AtomicU64::fetch_and).
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Within bounds
	/// assert_eq!(map.set(64, true, Relaxed), Some(false));
	///
	/// // Out of bounds
	/// assert_eq!(map.set(128, true, Relaxed), None);
	/// ```
	fn set(
		&self,
		idx: usize,
		val: bool,
		order: Ordering,
	) -> Option<bool> {
		let slot = self.slots().get(idx / SLOT_BITS)?;
		let mask = 1 << (idx & SLOT_MASK);
		let prev = if val {
			slot.fetch_or(mask, order)
		} else {
			slot.fetch_and(!mask, order)
		};
		Some(prev & mask != 0)
	}

	/// Identical to [`AtomicBitmap::set()`] but without any bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must guarantee that the index is within bounds of
	/// the bitmap.
	unsafe fn set_unchecked(
		&self,
		idx: usize,
		val: bool,
		order: Ordering,
	) -> bool {
		let slot = self.slots().get_unchecked(idx / SLOT_BITS);
		let mask = 1 << (idx & SLOT_MASK);
		let prev = if val {
			slot.fetch_or(mask, order)
		} else {
			slot.fetch_and(!mask, order)
		};
		prev & mask != 0
	}

	/// Checks whether the given bit index is within bounds of the
	/// bitmap.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	///
	/// let map = FixedBitmap::<4>::new(false);
	/// assert!(map.valid_index(64));
	/// assert!(map.valid_index(128));
	/// assert!(map.valid_index(255));
	/// assert!(!map.valid_index(256));
	/// ```
	fn valid_index(&self, idx: usize) -> bool {
		idx / SLOT_BITS < self.slots().len()
	}

	/// Inverts the bit at the given index. Returns the previous value
	/// of the bit, or [`None`] if the index is out of bounds.
	///
	/// The memory ordering corresponds to that of
	/// [`fetch_xor()`](AtomicU64::fetch_xor) and
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<1>::new(false);
	/// assert_eq!(map.flip(4, Relaxed), Some(false));
	/// assert_eq!(map.get(4, Relaxed), Some(true));
	/// ```
	fn flip(&self, idx: usize, order: Ordering) -> Option<bool> {
		let slot = self.slots().get(idx / SLOT_BITS)?;
		let mask = 1 << (idx & SLOT_MASK);
		let prev = slot.fetch_xor(mask, order);
		Some(prev & mask != 0)
	}

	/// Atomically swap two bits within the same 64-bit block. Returns
	/// [`None`] if the bits are in different blocks or if at least
	/// one of the indexes is out of bounds.
	///
	/// `order_a` corresponds to the ordering for the first load of
	/// the 64-bit block. `order_b` corresponds to the ordering for
	/// the successful compare-exchange of the block with the bits
	/// swapped.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Set some bit to 1, then swap it
	/// map.set(62, true, Relaxed).unwrap();
	/// map.swap(4, 62, Relaxed, Relaxed).unwrap();
	///
	/// assert_eq!(map.get(62, Relaxed), Some(false));
	/// assert_eq!(map.get(4, Relaxed), Some(true));
	/// assert_eq!(map.set_bits(Relaxed), 1);
	///
	/// // Cannot swap bits in different 64-bit blocks.
	/// assert_eq!(map.swap(4, 68, Relaxed, Relaxed), None);
	/// ```
	fn swap(
		&self,
		a: usize,
		b: usize,
		order_a: Ordering,
		order_b: Ordering,
	) -> Option<()> {
		let idx = a / SLOT_BITS;
		if idx != b / SLOT_BITS {
			// Different block, cannot swap atomically
			return None;
		}

		let slot = self.slots().get(idx)?;

		let bit_a = a & SLOT_MASK;
		let bit_b = b & SLOT_MASK;
		if bit_a == bit_b {
			// Nothing to do
			return Some(());
		}

		let mut value = slot.load(order_a);
		loop {
			let tmp = (value >> bit_a) ^ (value >> bit_b) & 1;
			let new = value ^ ((tmp << bit_a) | (tmp << bit_b));
			let Err(cur) = slot.compare_exchange_weak(
				value,
				new,
				order_b,
				Ordering::Relaxed,
			) else {
				return Some(());
			};

			value = cur;
		}
	}

	/// Returns a [`Some<Ok>`] with the previous value of the bit if
	/// the swap succeeded, or [`Some<Err>`] with the current value
	/// of the bit if it failed. Returns [`None`] if the index is out
	/// of bounds.
	///
	/// The operation will fail if the value of the bit does not match
	/// `current`, or if another thread modified different bit in the
	/// same block during the operation.
	///
	/// `load_order` corresponds to the first load of the 64-bit
	/// block. `success` and `failure` correspond to the orderings
	/// of [`compare_exchange()`](AtomicU64::compare_exchange).
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<4>::new(false);
	///
	/// // Swap success, bit was 0, now it's 1
	/// assert_eq!(
	/// 	map.compare_exchange(
	/// 		42, false, true, Relaxed, Relaxed, Relaxed
	/// 	),
	/// 	Some(Ok(false))
	/// );
	///
	/// // Swap failure, bit was 1
	/// assert_eq!(
	/// 	map.compare_exchange(
	/// 		42, false, true, Relaxed, Relaxed, Relaxed
	/// 	),
	/// 	Some(Err(true))
	/// );
	///
	/// // Out of bounds
	/// assert_eq!(
	/// 	map.compare_exchange(
	/// 		257, false, true, Relaxed, Relaxed, Relaxed
	/// 	),
	/// 	None
	/// );
	/// ```
	fn compare_exchange(
		&self,
		idx: usize,
		current: bool,
		new: bool,
		load_order: Ordering,
		success: Ordering,
		failure: Ordering,
	) -> Option<Result<bool, bool>> {
		let slot = self.slots().get(idx / SLOT_BITS)?;
		let mask = 1 << (idx & SLOT_MASK);
		let value = slot.load(load_order);

		let cur = if current { value | mask } else { value & !mask };
		let new = if new { value | mask } else { value & !mask };

		Some(
			slot.compare_exchange(cur, new, success, failure)
				.map(|val| val & mask != 0)
				.map_err(|val| val & mask != 0),
		)
	}

	/// Identical to [`AtomicBitmap::compare_exchange()`], except the
	/// function is allowed to fail spuriously (e.g. for another
	/// reason other than the ones listed in for `compare_exchange`).
	/// Check the documentation for
	/// [`AtomicU64::compare_exchange_weak()`] for a more complete
	/// explaination.
	fn compare_exchange_weak(
		&self,
		idx: usize,
		current: bool,
		new: bool,
		load_order: Ordering,
		success: Ordering,
		failure: Ordering,
	) -> Option<Result<bool, bool>> {
		let slot = self.slots().get(idx / SLOT_BITS)?;
		let mask = 1 << (idx & SLOT_MASK);
		let value = slot.load(load_order);

		let cur = if current { value | mask } else { value & !mask };
		let new = if new { value | mask } else { value & !mask };

		Some(
			slot.compare_exchange_weak(cur, new, success, failure)
				.map(|val| val & mask != 0)
				.map_err(|val| val & mask != 0),
		)
	}

	/// Atomically gets the index of the lowest zero bit in the
	/// bitmap, or [`None`] if all the bits are set.
	///
	/// `order` specifies the memory ordering when loading each
	/// 64-bit block.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<4>::new(true);
	/// assert_eq!(map.lowest_zero(Relaxed), None);
	///
	/// // Set some bit to 0
	/// map.set(60, false, Relaxed).unwrap();
	/// assert_eq!(map.lowest_zero(Relaxed), Some(60));
	///
	/// // Set a lower bit to 0
	/// map.set(4, false, Relaxed).unwrap();
	/// assert_eq!(map.lowest_zero(Relaxed), Some(4));
	/// ```
	fn lowest_zero(&self, order: Ordering) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let bit = slot.load(order).trailing_ones() as usize;
			if bit < SLOT_BITS {
				return Some(i * SLOT_BITS + bit);
			}
		}
		None
	}

	/// Atomically finds the lowest bit set to 0 and sets it to
	/// one. Returns the index of that bit, or [`None`] if all the
	/// bits are set.
	///
	/// `order_a` corresponds to the ordering for the first load of
	/// the 64-bit block. `order_b` corresponds to the ordering for
	/// the successful [compare-exchange](AtomicU64::compare_exchange)
	/// of the block with the bit set.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<4>::new(true);
	///
	/// // Set some bit to 0
	/// map.set(60, false, Relaxed).unwrap();
	/// assert_eq!(map.get(60, Relaxed), Some(false));
	///
	/// // Atomically set it to 1
	/// assert_eq!(map.set_lowest_zero(Relaxed, Relaxed), Some(60));
	///
	/// // No zero bits left
	/// assert_eq!(map.set_lowest_zero(Relaxed, Relaxed), None);
	/// ```
	fn set_lowest_zero(
		&self,
		order_a: Ordering,
		order_b: Ordering,
	) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let mut value = slot.load(order_a);

			loop {
				let bit = value.trailing_ones() as usize;
				if bit >= SLOT_BITS {
					// Slot is full, go on to the next one.
					break;
				}
				let new = value | (1 << bit);

				let Err(cur) = slot.compare_exchange_weak(
					value,
					new,
					order_b,
					Ordering::Relaxed,
				) else {
					// Success, return the set bit
					return Some(i * SLOT_BITS + bit);
				};

				value = cur;
			}
		}

		None
	}

	/// Similarly to [`AtomicBitmap::set_lowest_zero()`], atomically
	/// finds the lowest bit set to 0 and sets it to 1. The main
	/// difference is that this function only tries to set an unset
	/// bit in each slot once, which might skip over unset bits in
	/// case there is contention on the same slot. This means this
	/// function has better performance, but it might not find an
	/// unset bit in some cases (returning [`None`]), or the found
	/// bit might not be the lowest available.
	///
	/// `order_a` corresponds to the ordering for the first load of
	/// the 64-bit block. `order_b` corresponds to the ordering for
	/// the successful [compare-exchange](AtomicU64::compare_exchange)
	/// of the block with the bit set.
	fn set_lowest_zero_weak(
		&self,
		order_a: Ordering,
		order_b: Ordering,
	) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let value = slot.load(order_a);
			let bit = value.trailing_ones() as usize;
			if bit >= SLOT_BITS {
				continue;
			}
			let new = value | (1 << bit);

			// We don't care about the actual value of the block if
			// the comparison fails, so the load ordering in that
			// case can be relaxed.
			if slot
				.compare_exchange(
					value,
					new,
					order_b,
					Ordering::Relaxed,
				)
				.is_ok()
			{
				return Some(i * SLOT_BITS + bit);
			}
		}

		None
	}

	/// Atomically gets the index of the lowest set bit in the
	/// bitmap, or [`None`] if all the bits are unset.
	///
	/// `order` specifies the memory ordering when loading each
	/// 64-bit block.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<4>::new(false);
	/// assert_eq!(map.lowest_one(Relaxed), None);
	///
	/// // Set some bit to 1
	/// map.set(60, true, Relaxed).unwrap();
	/// assert_eq!(map.lowest_one(Relaxed), Some(60));
	///
	/// // Set a lower bit to 1
	/// map.set(4, true, Relaxed).unwrap();
	/// assert_eq!(map.lowest_one(Relaxed), Some(4));
	/// ```
	fn lowest_one(&self, order: Ordering) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let bit = slot.load(order).trailing_zeros() as usize;
			if bit < SLOT_BITS {
				return Some(i * SLOT_BITS + bit);
			}
		}
		None
	}

	/// Atomically finds the lowest bit set to 1 and sets it to
	/// zero. Returns the index of that bit, or [`None`] if all the
	/// bits are unset.
	///
	/// `order_a` corresponds to the ordering for the first load of
	/// the 64-bit block. `order_b` corresponds to the ordering for
	/// the successful [compare-exchange](AtomicU64::compare_exchange)
	/// of the block with the bit unset.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<4>::new(false);
	///
	/// // Set some bit to 1
	/// map.set(60, true, Relaxed).unwrap();
	/// assert_eq!(map.get(60, Relaxed), Some(true));
	///
	/// // Atomically set it to 0
	/// assert_eq!(map.clear_lowest_one(Relaxed, Relaxed), Some(60));
	///
	/// // No set bits left
	/// assert_eq!(map.clear_lowest_one(Relaxed, Relaxed), None);
	/// ```
	fn clear_lowest_one(
		&self,
		order_a: Ordering,
		order_b: Ordering,
	) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let mut value = slot.load(order_a);

			loop {
				let bit = value.trailing_zeros() as usize;
				if bit >= SLOT_BITS {
					// Slot is full, go on to the next one.
					break;
				}
				let new = value & !(1 << bit);

				let Err(cur) = slot.compare_exchange_weak(
					value,
					new,
					order_b,
					Ordering::Relaxed,
				) else {
					// Success, return the cleared bit
					return Some(i * SLOT_BITS + bit);
				};

				value = cur;
			}
		}

		None
	}

	/// Similarly to [`AtomicBitmap::clear_lowest_one()`], atomically
	/// finds the lowest bit set to 1 and sets it to 0. The main
	/// difference is that this function only tries to set a set bit
	/// in each slot once, which might skip over set bits in case
	/// there is contention on the same slot. This means this
	/// function has better performance, but it might not find a set
	/// bit in some cases (returning [`None`]), or the found bit
	/// might not be the lowest available.
	///
	/// `order_a` corresponds to the ordering for the first load of
	/// the 64-bit block. `order_b` corresponds to the ordering for
	/// the successful [compare-exchange](AtomicU64::compare_exchange)
	/// of the block with the bit unset.
	fn clear_lowest_one_weak(
		&self,
		order_a: Ordering,
		order_b: Ordering,
	) -> Option<usize> {
		for (i, slot) in self.slots().iter().enumerate() {
			let value = slot.load(order_a);
			let bit = value.trailing_zeros() as usize;
			if bit >= SLOT_BITS {
				continue;
			}
			let new = value & !(1 << bit);

			// We don't care about the actual value of the block if
			// the comparison fails, so the load ordering in that
			// case can be relaxed.
			if slot
				.compare_exchange(
					value,
					new,
					order_b,
					Ordering::Relaxed,
				)
				.is_ok()
			{
				return Some(i * SLOT_BITS + bit);
			}
		}

		None
	}

	/// Count the number of bits set to 1. This is not guaranteed to
	/// be an exact count if `N` > 1 because blocks can only be
	/// atomically read one at a time, which means changes can happen
	/// between the load of each 64-bit block.
	///
	/// `order` corresponds to the ordering for the load of each block
	/// when counting set bits.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<1>::new(true);
	/// assert_eq!(map.set_bits(Relaxed), 64);
	///
	/// // Set a single bit to 0.
	/// map.set(4, false, Relaxed);
	/// assert_eq!(map.set_bits(Relaxed), 63);
	/// ```
	fn set_bits(&self, order: Ordering) -> usize {
		self.slots()
			.iter()
			.map(|c| c.load(order).count_ones() as usize)
			.sum()
	}

	/// Reset all bits to the given value.
	///
	/// `order` corresponds to the ordering for the store of each
	/// block.
	///
	/// # Safety
	///
	/// This function is not fully atomic if there is more than one
	/// slot. In that case, the caller must guarantee that no
	/// concurrent reads or writes are happening on the bitmap, or
	/// that it is fine for the application to observe an
	/// intermediate state where only some blocks have been reset.
	///
	/// # Example
	/// ```rust
	/// use atomic_bitmap::{AtomicBitmap, FixedBitmap};
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<8>::new(true);
	/// assert_eq!(map.get(0, Relaxed), Some(true));
	///
	/// // SAFETY: no concurrent reads or writes are being made.
	/// unsafe { map.reset(false, Relaxed) };
	/// assert_eq!(map.get(0, Relaxed), Some(false));
	/// ```
	unsafe fn reset(&self, val: bool, order: Ordering) {
		let val = if val { u64::MAX } else { 0 };
		for slot in self.slots().iter() {
			slot.store(val, order);
		}
	}
}

/// A statically-sized atomic bitmap with a guaranteed non-zero
/// capacity.
///
/// The non-zero capacity guarantee is enforced at compile time:
///
/// ```compile_fail
/// use atomic_bitmap::FixedBitmap;
/// # // Miri does not properly run our compile-time assert.
/// # #[cfg(miri)]
/// # core::compile_error!("Miri");
///
/// // This will not build.
/// let bitmap = FixedBitmap::<0>::new(true);
/// ```
#[derive(Debug)]
pub struct FixedBitmap<const N: usize> {
	slots: [AtomicU64; N],
}

impl<const N: usize> FixedBitmap<N> {
	// A hack to get a compile-time assertion
	const SIZE_OK: () =
		assert!(N > 0, "Cannot create an empty bitmap");

	/// A new bitmap with all bits set to the specified value.
	pub fn new(val: bool) -> Self {
		#[allow(clippy::let_unit_value)]
		let _ = Self::SIZE_OK;

		let val = if val { u64::MAX } else { 0 };
		Self {
			slots: array::from_fn(|_| AtomicU64::new(val)),
		}
	}

	/// Creates a copy of the bitmap.
	///
	/// `order` corresponds to the memory ordering for the load of
	/// each 64-bit block.
	///
	/// # Safety
	///
	/// This function is not fully atomic if `N` > 1. If this is the
	/// case, the caller must guarantee that no concurrent
	/// modifications are happening on the bitmap, or that it is fine
	/// for the application to get a partially inconsistent view of
	/// the bitmap. This is because blocks can only be atomically
	/// read one at a time, which means changes can happen between
	/// the load of each 64-bit block.
	///
	/// Each 64-bit block is guaranteed to be internally consistent.
	/// Even when getting an inconsistent view of the bitmap, this
	/// function does not produce undefined behavior.
	pub unsafe fn clone(&self, order: Ordering) -> Self {
		Self {
			slots: array::from_fn(|i| {
				let val = self.slots[i].load(order);
				AtomicU64::new(val)
			}),
		}
	}

	/// Get the bitmap as an array of [`u64`]'s without consuming the
	/// bitmap.
	///
	/// `order` corresponds to the memory ordering for the load of
	/// each 64-bit block.
	///
	/// # Safety
	///
	/// Like [`FixedBitmap::clone()`], this function is not fully
	/// atomic if `N` > 1, meaning that the caller must guarantee
	/// that no concurrent modifications are happening on the bitmap,
	/// or that it is fine for the application to get a partially
	/// inconsistent view of the bitmap.
	///
	/// Again, each 64-bit block is guaranteed to be internally
	/// consistent, and this function cannot produce undefined
	/// behavior.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	/// use core::sync::atomic::Ordering::*;
	///
	/// let map = FixedBitmap::<8>::new(true);
	///
	/// // SAFETY: no concurrent writes are being made.
	/// let inner = unsafe { map.get_inner(Relaxed) };
	/// assert_eq!(inner, [u64::MAX; 8]);
	///
	/// // Convert back
	/// let map = FixedBitmap::from(inner);
	/// ```
	pub unsafe fn get_inner(&self, order: Ordering) -> [u64; N] {
		array::from_fn(|i| self.slots[i].load(order))
	}

	/// Get the bitmap as an array of [`u64`]'s, consuming the bitmap.
	/// This is safe because passing self by value guarantees that no
	/// other threads are concurrently accessing the atomic data.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<8>::new(true);
	///
	/// let inner = map.into_inner();
	/// assert_eq!(inner, [u64::MAX; 8]);
	///
	/// // Convert back
	/// let map = FixedBitmap::from(inner);
	/// ```
	pub fn into_inner(self) -> [u64; N] {
		self.slots.map(|s| s.into_inner())
	}

	/// The total number of bits that this bitmap can hold.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(true);
	/// assert_eq!(map.capacity().get(), 64 * 4);
	/// ```
	pub const fn capacity(&self) -> NonZeroUsize {
		// This is safe because `N` is guaranteed to be greater than
		// zero at compile time. The multiplication can only wrap
		// into 0 if `N` is a huge number, at which point no system
		// will have the memory to hold such a bitmap.
		unsafe { NonZeroUsize::new_unchecked(SLOT_BITS * N) }
	}
}

impl<const N: usize> AtomicBitmap for FixedBitmap<N> {
	fn slots(&self) -> &[AtomicU64] {
		&self.slots
	}
}

impl<const N: usize> From<[u64; N]> for FixedBitmap<N> {
	fn from(values: [u64; N]) -> Self {
		Self {
			slots: array::from_fn(|i| AtomicU64::new(values[i])),
		}
	}
}

impl<const N: usize> From<&[u64; N]> for FixedBitmap<N> {
	fn from(values: &[u64; N]) -> Self {
		Self {
			slots: array::from_fn(|i| AtomicU64::new(values[i])),
		}
	}
}

impl<const N: usize> TryFrom<&[u64]> for FixedBitmap<N> {
	type Error = core::array::TryFromSliceError;

	fn try_from(values: &[u64]) -> Result<Self, Self::Error> {
		let arr: &[u64; N] = values.try_into()?;
		Ok(Self {
			slots: array::from_fn(|i| AtomicU64::new(arr[i])),
		})
	}
}

impl<const N: usize> From<[AtomicU64; N]> for FixedBitmap<N> {
	/// Construct a bitmap from its interior representation.
	fn from(values: [AtomicU64; N]) -> Self {
		Self { slots: values }
	}
}

impl<const N: usize> From<FixedBitmap<N>> for [AtomicU64; N] {
	/// Gets the interior representation of the bitmap.
	fn from(bitmap: FixedBitmap<N>) -> Self {
		bitmap.slots
	}
}

impl<const N: usize> Default for FixedBitmap<N> {
	/// A new bitmap with all bits set to 0.
	fn default() -> Self {
		Self::new(false)
	}
}

impl<T> AtomicBitmap for T
where
	T: AsRef<[AtomicU64]>,
{
	fn slots(&self) -> &[AtomicU64] {
		self.as_ref()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use core::sync::atomic::Ordering::*;

	#[test]
	fn test_set_get() {
		let bm = FixedBitmap::<8>::new(false);
		let prev = bm.set(4, true, Relaxed);
		assert_eq!(prev, Some(false));
		let val = bm.get(4, Relaxed);
		assert_eq!(val, Some(true));
	}

	#[test]
	fn test_lowest_zero() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.lowest_zero(Relaxed), None);

		let prev = bm.set(67, false, Relaxed);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(Relaxed), Some(67));

		let prev = bm.set(62, false, Relaxed);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(Relaxed), Some(62));

		let prev = bm.set(62, true, Relaxed);
		assert_eq!(prev, Some(false));
		assert_eq!(bm.lowest_zero(Relaxed), Some(67));
	}

	#[test]
	fn test_set_lowest_zero() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.lowest_zero(Relaxed), None);

		let prev = bm.set(67, false, Relaxed);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(Relaxed), Some(67));

		let prev = bm.set(62, false, Relaxed);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(Relaxed), Some(62));

		let prev_lowest = bm.set_lowest_zero(Relaxed, Relaxed);
		assert_eq!(prev_lowest, Some(62));
		assert_eq!(bm.get(62, Relaxed), Some(true));
	}

	#[test]
	fn test_set_bits() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.set_bits(Relaxed), 8 * 64);

		bm.set(67, false, Relaxed).unwrap();
		assert_eq!(bm.set_bits(Relaxed), 8 * 64 - 1);
	}

	#[test]
	fn lowest_zero_thread() {
		use std::thread;
		const NTHREAD: usize = 8;
		const PERTHREAD: usize = 16;
		const NUM_BITS_SET: usize = NTHREAD * PERTHREAD;

		let bm = FixedBitmap::<8>::new(false);
		let cap = bm.capacity().get();
		thread::scope(|s| {
			for _ in 0..NTHREAD {
				s.spawn(|| {
					for _ in 0..PERTHREAD {
						bm.set_lowest_zero(Acquire, SeqCst);
					}
				});
			}
		});

		for i in 0..NUM_BITS_SET {
			assert_eq!(bm.get(i, Relaxed), Some(true));
		}
		for i in NUM_BITS_SET..cap {
			assert_eq!(bm.get(i, Relaxed), Some(false));
		}
	}
}
