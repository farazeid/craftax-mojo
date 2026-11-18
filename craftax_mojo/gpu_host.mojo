# Host-side driver for the GPU pipeline: owns the device buffers and enqueues
# the step / reset / obs kernels.

from max.gpu.host import DeviceContext, DeviceBuffer

from .constants import *
from .state import NUM_I, NUM_F
from . import vec
from . import gpu


struct GpuBatch:
    var ctx: DeviceContext
    var n: Int
    var block: Int
    var reset_grid: Int
    var ints: DeviceBuffer[DType.int32]
    var floats: DeviceBuffer[DType.float32]
    var map: DeviceBuffer[DType.uint8]
    var mob: DeviceBuffer[DType.uint8]
    var rng: DeviceBuffer[DType.uint64]
    var obs: DeviceBuffer[DType.float32]
    var reward: DeviceBuffer[DType.float32]
    var done: DeviceBuffer[DType.uint8]
    var done_idx: DeviceBuffer[DType.int32]
    var done_count: DeviceBuffer[DType.int32]
    var scratch: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, n: Int, block: Int = 128, reset_grid: Int = 256
    ) raises:
        self.ctx = ctx
        self.n = n
        self.block = block
        self.reset_grid = reset_grid
        self.ints = ctx.enqueue_create_buffer[DType.int32](n * NUM_I)
        self.floats = ctx.enqueue_create_buffer[DType.float32](n * NUM_F)
        self.map = ctx.enqueue_create_buffer[DType.uint8](n * MAP_CELLS)
        self.mob = ctx.enqueue_create_buffer[DType.uint8](n * MAP_CELLS)
        self.rng = ctx.enqueue_create_buffer[DType.uint64](n * 2)
        self.obs = ctx.enqueue_create_buffer[DType.float32](n * OBS_SIZE)
        self.reward = ctx.enqueue_create_buffer[DType.float32](n)
        self.done = ctx.enqueue_create_buffer[DType.uint8](n)
        self.done_idx = ctx.enqueue_create_buffer[DType.int32](n)
        self.done_count = ctx.enqueue_create_buffer[DType.int32](1)
        self.scratch = ctx.enqueue_create_buffer[DType.float32](
            reset_grid * gpu.SCRATCH_PER_BLOCK
        )

    def _grid(self, threads: Int) -> Int:
        return (threads + self.block - 1) // self.block

    def _reset(self, seed: UInt64, all_envs: Bool) raises:
        self.ctx.enqueue_function[gpu.reset_kernel](
            self.ints, self.floats, self.map, self.mob, self.rng, self.obs,
            self.reward, self.done, Int32(self.n), self.done_idx,
            self.done_count, self.scratch, seed, Int32(1) if all_envs else Int32(0),
            grid_dim=self.reset_grid, block_dim=gpu.RESET_BLOCK,
        )

    def reset(self, seed: UInt64) raises:
        self._reset(seed, True)

    def step_random(self) raises:
        """One auto-reset step of every env under a uniform random policy."""
        self.done_count.enqueue_fill(0)
        self.ctx.enqueue_function[gpu.step_kernel](
            self.ints, self.floats, self.map, self.mob, self.rng, self.obs,
            self.reward, self.done, Int32(self.n), self.done_idx, self.done_count,
            grid_dim=self._grid(self.n), block_dim=self.block,
        )
        self._reset(0, False)

    def step_random_naive(self) raises:
        """Single fused kernel with in-thread resets (for comparison)."""
        self.ctx.enqueue_function[vec.step_random_kernel](
            self.ints, self.floats, self.map, self.mob, self.rng, self.obs,
            self.reward, self.done, Int32(self.n),
            grid_dim=self._grid(self.n), block_dim=self.block,
        )

    def synchronize(self) raises:
        self.ctx.synchronize()
