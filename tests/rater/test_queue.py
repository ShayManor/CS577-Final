from rater.queue import build_items, order_items


def test_items_three_per_clip(records):
    items = build_items(records)
    assert len(items) == 60
    r = records[0]
    mine = [i for i in items if i.clip_id == r.clip_id]
    assert [i.item_id for i in mine] == [f"{r.clip_id}:{k}" for k in range(3)]
    assert [i.instruction for i in mine] == [r.instruction, *r.alt_instructions]


def test_order_min_gap(records):
    items = build_items(records)
    order = order_items(items)
    assert sorted(i.item_id for i in order) == sorted(i.item_id for i in items)
    pos = {}
    for n, it in enumerate(order):
        if it.clip_id in pos:
            assert n - pos[it.clip_id] >= 5
        pos[it.clip_id] = n
    assert [i.item_id for i in order] == [i.item_id for i in order_items(items)]
    assert [i.item_id for i in order] != [i.item_id for i in order_items(items, seed=1)]


def test_order_min_gap_decision():
    from tests.rater.conftest import make_record
    records = [make_record(i) for i in range(51)]
    items = build_items(records)
    assert len(items) == 153 and len({i.group for i in items}) == 17
    order = order_items(items)
    pos = {}
    for n, it in enumerate(order):
        if it.group in pos:
            assert n - pos[it.group] >= 5
        pos[it.group] = n
