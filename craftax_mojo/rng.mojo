# Random number sources.
#
# The game logic draws randomness in exactly the order the JAX implementation
# does, through this interface. Two implementations:
#
# * `SplitMix`: counter-based splitmix64 used for benchmarking. Being
#   counter-based, an array draw of n values is just a reserved range of the
#   counter, so world generation can read noise/ore samples in any order.
# * `Tape`: replays the values JAX actually drew (recorded by
#   parity/record_jax.py), which lets us compare the Mojo port against JAX
#   transition-by-transition with identical randomness.
#
# `choice` returns an index. For choices over a masked set (mob spawn cells,
# diamond placement) the native generator samples the k-th valid cell, while
# the tape returns the flat index JAX picked; `IS_TAPE` tells callers which.

from std.memory import Pointer


trait Rng:
    comptime IS_TAPE: Bool

    def uniform(mut self) -> Float32:
        """One U[0,1) draw (jax.random.uniform with shape ())."""
        ...

    def choice(mut self, n: Int) -> Int:
        """One index in [0, n) (jax.random.choice / its recorded result)."""
        ...

    def skip(mut self):
        """A draw JAX makes whose value our caller does not need."""
        ...

    def reserve(mut self, n: Int) -> Int:
        """Reserve an array draw of n values; returns a base for `at`."""
        ...

    def at(self, base: Int, i: Int) -> Float32:
        """Element i of an array draw reserved with `reserve`."""
        ...


@inline(.always)
def _mix64(x: UInt64) -> UInt64:
    var z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


@inline(.always)
def seed_key(seed: UInt64, env: UInt64) -> UInt64:
    return _mix64(_mix64(seed + 0x9E3779B97F4A7C15) ^ (env * 0xD1B54A32D192ED03))


struct SplitMix(Rng):
    comptime IS_TAPE = False
    var key: UInt64
    var ctr: UInt64

    def __init__(out self, key: UInt64, ctr: UInt64):
        self.key = key
        self.ctr = ctr

    @inline(.always)
    def _bits(self, i: UInt64) -> UInt64:
        return _mix64(self.key + i * 0x9E3779B97F4A7C15)

    @inline(.always)
    def uniform(mut self) -> Float32:
        var b = self._bits(self.ctr)
        self.ctr += 1
        return Float32(b >> 40) * Float32(1.0 / 16777216.0)

    @inline(.always)
    def choice(mut self, n: Int) -> Int:
        var b = self._bits(self.ctr)
        self.ctr += 1
        # Lemire multiply-shift on the top 32 bits.
        return Int(((b >> 32) * UInt64(n)) >> 32)

    @inline(.always)
    def skip(mut self):
        pass

    @inline(.always)
    def reserve(mut self, n: Int) -> Int:
        var base = self.ctr
        self.ctr += UInt64(n)
        return Int(base)

    @inline(.always)
    def at(self, base: Int, i: Int) -> Float32:
        var b = self._bits(UInt64(base + i))
        return Float32(b >> 40) * Float32(1.0 / 16777216.0)


struct Tape(Rng):
    comptime IS_TAPE = True
    var data: Pointer[Float32, MutUntrackedOrigin]
    var pos: Int

    def __init__(out self, data: Pointer[Float32, MutUntrackedOrigin], pos: Int):
        self.data = data
        self.pos = pos

    @inline(.always)
    def uniform(mut self) -> Float32:
        var v = self.data[unsafe_offset = self.pos]
        self.pos += 1
        return v

    @inline(.always)
    def choice(mut self, n: Int) -> Int:
        var v = Int(self.data[unsafe_offset = self.pos])
        self.pos += 1
        return v

    @inline(.always)
    def skip(mut self):
        self.pos += 1

    @inline(.always)
    def reserve(mut self, n: Int) -> Int:
        var base = self.pos
        self.pos += n
        return base

    @inline(.always)
    def at(self, base: Int, i: Int) -> Float32:
        return self.data[unsafe_offset = base + i]
