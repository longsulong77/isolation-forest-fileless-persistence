from Evtx.Evtx import Evtx

path = r"C:\SysmonLab\data\raw\attack_run3.evtx"
with Evtx(path) as log:
    count = sum(1 for _ in log.records())
print(count, "records")