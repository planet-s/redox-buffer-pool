# `redox-buffer-pool`
A small library that provides a general-purpose O(log n)-for-the-most-part
memory allocator, suitable for small (32-bit at the moment) and possibly
short-lived allocations. It also comes with a "guarding" mechanism, which will
prevent the pool from reclaiming memory if it's in use by someone else.
