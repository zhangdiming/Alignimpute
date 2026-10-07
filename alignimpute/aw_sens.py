import sys

from . import linsync, sync

i = sys.argv.index("--")
aw = float(sys.argv[sys.argv.index("--aw") + 1])
sync.THIN_CFG["anchor_weight"] = aw
sys.argv = [sys.argv[0]] + sys.argv[i + 1:]
linsync.main()
