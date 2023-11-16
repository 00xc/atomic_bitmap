#![cfg_attr(not(test), no_std)]

//! A statically sized atomic bitmap for concurrent systems.
//!
//! This crate performs no heap allocations and uses `#[no_std]`.
//!
//! # Example
//!
//! ```rust
//! use atomic_bitmap::FixedBitmap;
//!
//! // A bitmap with 128 bits, all set to zero.
//! let map = FixedBitmap::<2>::new(false);
//!
//! // Setting a bit and getting its previous value
//! assert_eq!(map.set(4, true), Some(false));
//!
//! // Setting an invalid bit
//! assert_eq!(map.set(128, true), None);
//!
//! // Conditionally setting a bit
//! assert_eq!(
//!     map.compare_exchange(102, false, true),
//!     Some(Ok(false)),
//! );
//!
//! // Getting the value of a bit
//! assert_eq!(map.get(4), Some(true));
//!
//! // Getting the value of an invalid bit
//! assert_eq!(map.get(128), None);
//!
//! // Clearing the lowest set bit
//! assert_eq!(map.clear_lowest_one(), Some(4));
//!
//! // Setting the lowest unset bit
//! assert_eq!(map.set_lowest_zero(), Some(0));
//! ```

use core::array;
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicU64, Ordering};

/// A statically-sized atomic bitmap.
///
/// The bitmap is generic over the number of 64-bit blocks it holds
/// (`N`). `N` is guaranteed to be greater than 0 at compile time:
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
	const SLOT_BITS: usize = u64::BITS as usize;
	const SLOT_MASK: usize = Self::SLOT_BITS - 1;

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

	/// Get the bit at the specified index, or [`None`] if the index
	/// is out of bounds.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Within bounds
	/// assert_eq!(map.get(64), Some(false));
	///
	/// // Out of bounds
	/// assert_eq!(map.get(128), None);
	/// ```
	pub fn get(&self, idx: usize) -> Option<bool> {
		let slot = self
			.slots
			.get(idx / Self::SLOT_BITS)?
			.load(Ordering::Acquire);
		let mask = 1 << (idx & Self::SLOT_MASK);
		Some(slot & mask != 0)
	}

	/// Identical to [`FixedBitmap::get()`] but without any bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must guarantee that the index is within bounds of
	/// the bitmap.
	pub unsafe fn get_unchecked(&self, idx: usize) -> bool {
		let slot = self
			.slots
			.get_unchecked(idx / Self::SLOT_BITS)
			.load(Ordering::Acquire);
		let mask = 1 << (idx & Self::SLOT_MASK);
		slot & mask != 0
	}

	/// Sets the specified bit to the given value and return the
	/// previous value, or [`None`] if the index is out of bounds.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Within bounds
	/// assert_eq!(map.set(64, true), Some(false));
	///
	/// // Out of bounds
	/// assert_eq!(map.set(128, true), None);
	/// ```
	pub fn set(&self, idx: usize, val: bool) -> Option<bool> {
		let slot = self.slots.get(idx / Self::SLOT_BITS)?;
		let mask = 1 << (idx & Self::SLOT_MASK);
		let prev = if val {
			slot.fetch_or(mask, Ordering::SeqCst)
		} else {
			slot.fetch_and(!mask, Ordering::SeqCst)
		};
		Some(prev & mask != 0)
	}

	/// Identical to [`FixedBitmap::set()`] but without any bounds
	/// checking.
	///
	/// # Safety
	///
	/// The caller must guarantee that the index is within bounds of
	/// the bitmap.
	pub unsafe fn set_unchecked(
		&self,
		idx: usize,
		val: bool,
	) -> bool {
		let slot = self.slots.get_unchecked(idx / Self::SLOT_BITS);
		let mask = 1 << (idx & Self::SLOT_MASK);
		let prev = if val {
			slot.fetch_or(mask, Ordering::SeqCst)
		} else {
			slot.fetch_and(!mask, Ordering::SeqCst)
		};
		prev & mask != 0
	}

	/// Checks whether the given bit index is within bounds of the
	/// bitmap.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(false);
	/// assert!(map.valid_index(64));
	/// assert!(map.valid_index(128));
	/// assert!(map.valid_index(255));
	/// assert!(!map.valid_index(256));
	/// ```
	pub const fn valid_index(&self, idx: usize) -> bool {
		idx / Self::SLOT_BITS < N
	}

	/// Inverts the bit at the given index. Returns the previous value
	/// of the bit, or [`None`] if the index is out of bounds.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<1>::new(false);
	/// assert_eq!(map.flip(4), Some(false));
	/// assert_eq!(map.get(4), Some(true));
	/// ```
	pub fn flip(&self, idx: usize) -> Option<bool> {
		let slot = self.slots.get(idx / Self::SLOT_BITS)?;
		let mask = 1 << (idx & Self::SLOT_MASK);
		let prev = slot.fetch_xor(mask, Ordering::SeqCst);
		Some(prev & mask != 0)
	}

	/// Atomically swap two bits within the same 64-bit block. Returns
	/// [`None`] if the bits are in different blocks or if at least
	/// one of the indexes is out of bounds.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<2>::new(false);
	///
	/// // Set some bit to 1, then swap it
	/// map.set(62, true).unwrap();
	/// map.swap(4, 62).unwrap();
	///
	/// assert_eq!(map.get(62), Some(false));
	/// assert_eq!(map.get(4), Some(true));
	/// assert_eq!(map.set_bits(), 1);
	///
	/// // Cannot swap bits in different 64-bit blocks.
	/// assert_eq!(map.swap(4, 68), None);
	/// ```
	pub fn swap(&self, a: usize, b: usize) -> Option<()> {
		let idx = a / Self::SLOT_BITS;
		if idx != b / Self::SLOT_BITS {
			// Different block, cannot swap atomically
			return None;
		}

		let slot = self.slots.get(idx)?;

		let bit_a = a & Self::SLOT_MASK;
		let bit_b = b & Self::SLOT_MASK;
		if bit_a == bit_b {
			// Nothing to do
			return Some(());
		}

		let mut value = slot.load(Ordering::Acquire);
		loop {
			let tmp = (value >> bit_a) ^ (value >> bit_b) & 1;
			let new = value ^ ((tmp << bit_a) | (tmp << bit_b));
			let Err(cur) = slot.compare_exchange_weak(
				value,
				new,
				Ordering::SeqCst,
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
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(false);
	///
	/// // Swap success, bit was 0, now it's 1
	/// assert_eq!(
	///     map.compare_exchange(42, false, true),
	///     Some(Ok(false))
	/// );
	///
	/// // Swap failure, bit was 1
	/// assert_eq!(
	///     map.compare_exchange(42, false, true),
	///     Some(Err(true))
	/// );
	///
	/// // Out of bounds
	/// assert_eq!(map.compare_exchange(257, false, true), None);
	/// ```
	pub fn compare_exchange(
		&self,
		idx: usize,
		current: bool,
		new: bool,
	) -> Option<Result<bool, bool>> {
		let slot = self.slots.get(idx / Self::SLOT_BITS)?;
		let mask = 1 << (idx & Self::SLOT_MASK);
		let value = slot.load(Ordering::Acquire);

		let cur = if current { value | mask } else { value & !mask };
		let new = if new { value | mask } else { value & !mask };

		Some(
			slot.compare_exchange(
				cur,
				new,
				Ordering::SeqCst,
				Ordering::SeqCst,
			)
			.map(|val| val & mask != 0)
			.map_err(|val| val & mask != 0),
		)
	}

	/// Identical to [`FixedBitmap::compare_exchange()`], except the
	/// function is allowed to fail spuriously (e.g. for another
	/// reason other than the ones listed in for `compare_exchange`).
	/// Check the documentation for
	/// [`AtomicU64::compare_exchange_weak()`] for a more complete
	/// explaination.
	pub fn compare_exchange_weak(
		&self,
		idx: usize,
		current: bool,
		new: bool,
	) -> Option<Result<bool, bool>> {
		let slot = self.slots.get(idx / Self::SLOT_BITS)?;
		let mask = 1 << (idx & Self::SLOT_MASK);
		let value = slot.load(Ordering::Acquire);

		let cur = if current { value | mask } else { value & !mask };
		let new = if new { value | mask } else { value & !mask };

		Some(
			slot.compare_exchange_weak(
				cur,
				new,
				Ordering::SeqCst,
				Ordering::SeqCst,
			)
			.map(|val| val & mask != 0)
			.map_err(|val| val & mask != 0),
		)
	}

	/// Atomically gets the index of the lowest zero bit in the
	/// bitmap, or [`None`] if all the bits are set.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(true);
	/// assert_eq!(map.lowest_zero(), None);
	///
	/// // Set some bit to zero
	/// map.set(60, false).unwrap();
	/// assert_eq!(map.lowest_zero(), Some(60));
	///
	/// // Set a lower bit to zero
	/// map.set(4, false).unwrap();
	/// assert_eq!(map.lowest_zero(), Some(4));
	/// ```
	pub fn lowest_zero(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let bit =
				slot.load(Ordering::Acquire).trailing_ones() as usize;
			if bit < Self::SLOT_BITS {
				return Some(i * Self::SLOT_BITS + bit);
			}
		}
		None
	}

	/// Atomically finds the lowest bit set to zero and sets it to
	/// one. Returns the index of that bit, or [`None`] if all the
	/// bits are set.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(true);
	///
	/// // Set some bit to zero
	/// map.set(60, false).unwrap();
	/// assert_eq!(map.get(60), Some(false));
	///
	/// // Atomically set it to one
	/// assert_eq!(map.set_lowest_zero(), Some(60));
	///
	/// // No zero bits left
	/// assert_eq!(map.set_lowest_zero(), None);
	/// ```
	pub fn set_lowest_zero(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let mut value = slot.load(Ordering::Acquire);

			loop {
				let bit = value.trailing_ones() as usize;
				if bit >= Self::SLOT_BITS {
					// Slot is full, go on to the next one.
					break;
				}
				let new = value | (1 << bit);

				let Err(cur) = slot.compare_exchange_weak(
					value,
					new,
					Ordering::SeqCst,
					Ordering::Relaxed,
				) else {
					// Success, return the set bit
					return Some(i * Self::SLOT_BITS + bit);
				};

				value = cur;
			}
		}

		None
	}

	/// Similarly to [`FixedBitmap::set_lowest_zero()`], atomically
	/// finds the lowest bit set to zero and sets it to one. The main
	/// difference is that this function only tries to set an unset
	/// bit in each slot once, which might skip over unset bits in
	/// case there is contention on the same slot. This means this
	/// function has better performance, but it might not find an
	/// unset bit in some cases (returning [`None`]), or the found
	/// bit might not be the lowest available.
	pub fn set_lowest_zero_weak(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let value = slot.load(Ordering::Acquire);
			let bit = value.trailing_ones() as usize;
			if bit >= Self::SLOT_BITS {
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
					Ordering::SeqCst,
					Ordering::Relaxed,
				)
				.is_ok()
			{
				return Some(i * Self::SLOT_BITS + bit);
			}
		}

		None
	}

	/// Atomically gets the index of the lowest set bit in the
	/// bitmap, or [`None`] if all the bits are unset.
	//
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(false);
	/// assert_eq!(map.lowest_one(), None);
	///
	/// // Set some bit to one
	/// map.set(60, true).unwrap();
	/// assert_eq!(map.lowest_one(), Some(60));
	///
	/// // Set a lower bit to one
	/// map.set(4, true).unwrap();
	/// assert_eq!(map.lowest_one(), Some(4));
	/// ```
	pub fn lowest_one(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let bit = slot.load(Ordering::Acquire).trailing_zeros()
				as usize;
			if bit < Self::SLOT_BITS {
				return Some(i * Self::SLOT_BITS + bit);
			}
		}
		None
	}

	/// Atomically finds the lowest bit set to one and sets it to
	/// zero. Returns the index of that bit, or [`None`] if all the
	/// bits are unset.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<4>::new(false);
	///
	/// // Set some bit to one
	/// map.set(60, true).unwrap();
	/// assert_eq!(map.get(60), Some(true));
	///
	/// // Atomically set it to zero
	/// assert_eq!(map.clear_lowest_one(), Some(60));
	///
	/// // No set bits left
	/// assert_eq!(map.clear_lowest_one(), None);
	/// ```
	pub fn clear_lowest_one(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let mut value = slot.load(Ordering::Acquire);

			loop {
				let bit = value.trailing_zeros() as usize;
				if bit >= Self::SLOT_BITS {
					// Slot is full, go on to the next one.
					break;
				}
				let new = value & !(1 << bit);

				let Err(cur) = slot.compare_exchange_weak(
					value,
					new,
					Ordering::SeqCst,
					Ordering::Relaxed,
				) else {
					// Success, return the cleared bit
					return Some(i * Self::SLOT_BITS + bit);
				};

				value = cur;
			}
		}

		None
	}

	/// Similarly to [`FixedBitmap::clear_lowest_one()`], atomically
	/// finds the lowest bit set to one and sets it to zero. The main
	/// difference is that this function only tries to set a set bit
	/// in each slot once, which might skip over set bits in case
	/// there is contention on the same slot. This means this
	/// function has better performance, but it might not find a set
	/// bit in some cases (returning [`None`]), or the found bit
	/// might not be the lowest available.
	pub fn clear_lowest_one_weak(&self) -> Option<usize> {
		for (i, slot) in self.slots.iter().enumerate() {
			let value = slot.load(Ordering::Acquire);
			let bit = value.trailing_zeros() as usize;
			if bit >= Self::SLOT_BITS {
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
					Ordering::SeqCst,
					Ordering::Relaxed,
				)
				.is_ok()
			{
				return Some(i * Self::SLOT_BITS + bit);
			}
		}

		None
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
		// into zero if `N` is a huge number, at which point no system
		// will have the memory to hold such a bitmap.
		unsafe { NonZeroUsize::new_unchecked(Self::SLOT_BITS * N) }
	}

	/// Count the number of bits set to 1. This is not guaranteed to
	/// be an exact count if `N` > 1 because blocks can only be
	/// atomically read one at a time, which means changes can happen
	/// between the load of each 64-bit block.
	///
	/// # Example
	///
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<1>::new(true);
	/// assert_eq!(map.set_bits(), 64);
	///
	/// // Set a single bit to zero.
	/// map.set(4, false);
	/// assert_eq!(map.set_bits(), 63);
	/// ```
	pub fn set_bits(&self) -> usize {
		self.slots
			.iter()
			.map(|c| c.load(Ordering::Acquire).count_ones() as usize)
			.sum()
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

	/// Creates a copy of the bitmap.
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
	pub unsafe fn clone(&self) -> Self {
		Self {
			slots: array::from_fn(|i| {
				let val = self.slots[i].load(Ordering::SeqCst);
				AtomicU64::new(val)
			}),
		}
	}

	/// Reset all bits to the given value.
	///
	/// # Safety
	///
	/// Like [`FixedBitmap::clone()`], this function is not fully
	/// atomic if `N` > 1. In that case, the caller must guarantee
	/// that no concurrent reads or writes are happening on the
	/// bitmap, or that it is fine for the application to observe an
	/// intermediate state where only some blocks have been reset.
	///
	/// # Example
	/// ```rust
	/// use atomic_bitmap::FixedBitmap;
	///
	/// let map = FixedBitmap::<8>::new(true);
	/// assert_eq!(map.get(0), Some(true));
	///
	/// // SAFETY: no concurrent reads or writes are being made.
	/// unsafe { map.reset(false) };
	/// assert_eq!(map.get(0), Some(false));
	/// ```
	pub unsafe fn reset(&self, val: bool) {
		let val = if val { u64::MAX } else { 0 };
		for slot in self.slots.iter() {
			slot.store(val, Ordering::Release);
		}
	}

	/// Get the bitmap as an array of [`u64`]'s without consuming the
	/// bitmap.
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
	///
	/// let map = FixedBitmap::<8>::new(true);
	///
	/// // SAFETY: no concurrent writes are being made.
	/// let inner = unsafe { map.get_inner() };
	/// assert_eq!(inner, [u64::MAX; 8]);
	///
	/// // Convert back
	/// let map = FixedBitmap::from(inner);
	/// ```
	pub unsafe fn get_inner(&self) -> [u64; N] {
		array::from_fn(|i| self.slots[i].load(Ordering::SeqCst))
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
	/// A new bitmap with all bits set to zero.
	fn default() -> Self {
		Self::new(false)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_set_get() {
		let bm = FixedBitmap::<8>::new(false);
		let prev = bm.set(4, true);
		assert_eq!(prev, Some(false));
		let val = bm.get(4);
		assert_eq!(val, Some(true));
	}

	#[test]
	fn test_lowest_zero() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.lowest_zero(), None);

		let prev = bm.set(67, false);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(), Some(67));

		let prev = bm.set(62, false);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(), Some(62));

		let prev = bm.set(62, true);
		assert_eq!(prev, Some(false));
		assert_eq!(bm.lowest_zero(), Some(67));
	}

	#[test]
	fn test_set_lowest_zero() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.lowest_zero(), None);

		let prev = bm.set(67, false);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(), Some(67));

		let prev = bm.set(62, false);
		assert_eq!(prev, Some(true));
		assert_eq!(bm.lowest_zero(), Some(62));

		let prev_lowest = bm.set_lowest_zero();
		assert_eq!(prev_lowest, Some(62));
		assert_eq!(bm.get(62), Some(true));
	}

	#[test]
	fn test_set_bits() {
		let bm = FixedBitmap::<8>::new(true);
		assert_eq!(bm.set_bits(), 8 * 64);

		bm.set(67, false).unwrap();
		assert_eq!(bm.set_bits(), 8 * 64 - 1);
	}
}
