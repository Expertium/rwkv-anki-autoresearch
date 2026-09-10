"""Spawn-mode check for get_result_cf.py: a worker must see the PATCHED insert_probes."""


def check(q):
    import rwkv.prepare_batch as p
    q.put(p.insert_probes.__name__)
