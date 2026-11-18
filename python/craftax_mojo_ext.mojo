# Python bindings. Arrays are owned by numpy and passed as raw addresses.
#
#   layout()                               -> dict of layout constants
#   reset_tape(state, tape)                -> draws consumed
#   step_tape(state, action, tape)         -> (reward, done, draws consumed)
#   write_obs(state, obs)                  -> None
#   reset_cpu(batch, seed, workers)        -> None
#   step_cpu(batch, actions, workers)      -> None
#   step_random_cpu(batch, workers)        -> None
#
# `state` is a tuple (ints, floats, map, mob_map) of addresses for a single
# env (env-major); `batch` is (ints, floats, map, mob_map, rng, obs, reward,
# done, n).

from std.memory import Pointer
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from craftax_mojo.constants import *
from craftax_mojo.state import *
from craftax_mojo.rng import Tape
from craftax_mojo.worldgen import generate_world
from craftax_mojo.game import step, is_game_over
from craftax_mojo.obs import write_obs as _write_obs
from craftax_mojo import vec


@inline(.always)
def _ptr[T: AnyType](o: PythonObject) raises -> Pointer[T, MutUntrackedOrigin]:
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(py=o))


def _env(state: PythonObject) raises -> EnvView[False]:
    return EnvView[False](
        _ptr[Int32](state[0]),
        _ptr[Float32](state[1]),
        _ptr[UInt8](state[2]),
        _ptr[UInt8](state[3]),
        1,
        0,
    )


def _batch(b: PythonObject) raises -> vec.Buffers:
    return vec.Buffers(
        _ptr[Int32](b[0]),
        _ptr[Float32](b[1]),
        _ptr[UInt8](b[2]),
        _ptr[UInt8](b[3]),
        _ptr[UInt64](b[4]),
        _ptr[Float32](b[5]),
        _ptr[Float32](b[6]),
        _ptr[UInt8](b[7]),
        Int(py=b[8]),
    )


def layout() raises -> PythonObject:
    var d = Python.dict()
    d["NUM_I"] = NUM_I
    d["NUM_F"] = NUM_F
    d["MAP_CELLS"] = MAP_CELLS
    d["OBS_SIZE"] = OBS_SIZE
    d["I_PX"] = I_PX
    d["I_PY"] = I_PY
    d["I_DIR"] = I_DIR
    d["I_HEALTH"] = I_HEALTH
    d["I_FOOD"] = I_FOOD
    d["I_DRINK"] = I_DRINK
    d["I_ENERGY"] = I_ENERGY
    d["I_SLEEPING"] = I_SLEEPING
    d["I_TIMESTEP"] = I_TIMESTEP
    d["I_INV"] = I_INV
    d["I_ZOMBIE"] = I_ZOMBIE
    d["I_COW"] = I_COW
    d["I_SKELETON"] = I_SKELETON
    d["I_ARROW"] = I_ARROW
    d["I_ARROW_DIR"] = I_ARROW_DIR
    d["I_PLANT_POS"] = I_PLANT_POS
    d["I_PLANT_AGE"] = I_PLANT_AGE
    d["I_PLANT_MASK"] = I_PLANT_MASK
    d["I_ACH"] = I_ACH
    d["F_RECOVER"] = F_RECOVER
    d["F_HUNGER"] = F_HUNGER
    d["F_THIRST"] = F_THIRST
    d["F_FATIGUE"] = F_FATIGUE
    d["F_LIGHT"] = F_LIGHT
    return d


def reset_tape(state: PythonObject, tape: PythonObject) raises -> PythonObject:
    var env = _env(state)
    var rng = Tape(_ptr[Float32](tape), 0)
    generate_world(env, rng)
    return rng.pos


def step_tape(
    state: PythonObject, action: PythonObject, tape: PythonObject
) raises -> PythonObject:
    var env = _env(state)
    var rng = Tape(_ptr[Float32](tape), 0)
    var r = step(env, Int(py=action), rng)
    var d = is_game_over(env)
    return Python.tuple(Float64(r), d, rng.pos)


def write_obs(state: PythonObject, obs: PythonObject) raises -> PythonObject:
    _write_obs(_env(state), _ptr[Float32](obs))
    return None


def reset_cpu(
    b: PythonObject, seed: PythonObject, workers: PythonObject
) raises -> PythonObject:
    vec.reset_cpu(_batch(b), UInt64(Int(py=seed)), Int(py=workers))
    return None


def step_cpu(
    b: PythonObject, actions: PythonObject, workers: PythonObject
) raises -> PythonObject:
    vec.step_cpu(_batch(b), _ptr[Int32](actions), Int(py=workers))
    return None


def step_random_cpu(b: PythonObject, workers: PythonObject) raises -> PythonObject:
    vec.step_random_cpu(_batch(b), Int(py=workers))
    return None


@export
def PyInit_craftax_mojo_ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("craftax_mojo_ext")
        m.def_function[layout]("layout")
        m.def_function[reset_tape]("reset_tape")
        m.def_function[step_tape]("step_tape")
        m.def_function[write_obs]("write_obs")
        m.def_function[reset_cpu]("reset_cpu")
        m.def_function[step_cpu]("step_cpu")
        m.def_function[step_random_cpu]("step_random_cpu")
        return m.finalize()
    except e:
        abort(t"failed to create craftax_mojo_ext: {e}")
