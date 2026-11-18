# Mojo PPO trainer entry point (see craftax_mojo/ppo.mojo).
#
#   mojo build -I . --fp-mode contract=off train/ppo_mojo.mojo -o build/ppo_mojo
#   ./build/ppo_mojo --total-steps 5e6 [--num-envs 512] [--seed 42]
#                    [--precision default|highest] [--out results/train/mojo.json]

from std.sys import argv

from craftax_mojo.ppo import Config, train


def main() raises:
    var cfg = Config()
    var out_path = String("")
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--total-steps":
            cfg.total_steps = Int(Float64(String(args[i + 1])))
        elif a == "--num-envs":
            cfg.num_envs = Int(String(args[i + 1]))
        elif a == "--seed":
            cfg.seed = UInt64(Int(String(args[i + 1])))
        elif a == "--precision":
            cfg.tf32 = String(args[i + 1]) != "highest"
        elif a == "--out":
            out_path = String(args[i + 1])
        else:
            raise Error("unknown argument " + a)
        i += 2
    train(cfg, out_path)
