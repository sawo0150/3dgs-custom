import unittest
from collections import Counter

from runtime.packet_scheduler import PacketIntervalRandomReshuffling
from runtime.scheduler import RelativeFloorIntervalSoftmaxRandomReshuffling


class PacketTests(unittest.TestCase):
    def test_exact_outer_pairing_with_arrivals(self):
        for seed in range(8):
            base = RelativeFloorIntervalSoftmaxRandomReshuffling(seed, 1.1, 8)
            packet = PacketIntervalRandomReshuffling(seed, 1.1, 8, 3, 2)
            next_id = 0
            for step in range(500):
                if step % 11 == 0:
                    ids = list(range(next_id, next_id + 7))
                    base.add(ids); packet.add(ids); next_id += 7
                a, b = base.draw(), packet.draw()
                self.assertEqual(base.frame_to_interval[a], packet.frame_to_interval[b])
                self.assertEqual(base.interval_counts, packet.interval_counts)
                self.assertLess(b, next_id)

    def test_repeats_one_exact_legacy(self):
        base = RelativeFloorIntervalSoftmaxRandomReshuffling(4, 1.1, 8)
        packet = PacketIntervalRandomReshuffling(4, 1.1, 8, 3, 1)
        for step in range(300):
            if step % 19 == 0:
                ids = range(step * 5, step * 5 + 5)
                base.add(ids); packet.add(ids)
            self.assertEqual(base.draw(), packet.draw())

    def test_ragged_cycles_and_count_bound(self):
        for n in (1, 3, 7, 8, 11):
            for size in (1, 4, 20):
                s = PacketIntervalRandomReshuffling(packet_size=size, repeats=2)
                s.add(range(n))
                for _ in range(3):
                    counts = Counter({i: 0 for i in range(n)})
                    for _ in range(n * 2):
                        counts[s.draw()] += 1
                        self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)
                    self.assertEqual(set(counts.values()), {2})

    def test_each_packet_pass_without_replacement(self):
        s = PacketIntervalRandomReshuffling(packet_size=4, repeats=2)
        s.add(range(8))
        draws = [s.draw() for _ in range(16)]
        self.assertEqual(set(draws[:4]), set(draws[4:8]))
        self.assertEqual(set(draws[8:12]), set(draws[12:]))
        self.assertTrue(set(draws[:4]).isdisjoint(draws[8:12]))
        self.assertTrue(all(len(set(draws[k:k+4])) == 4 for k in range(0, 16, 4)))

    def test_invalid_parameters(self):
        for size, repeats in ((0, 2), (4, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                PacketIntervalRandomReshuffling(packet_size=size, repeats=repeats)


if __name__ == "__main__":
    unittest.main()
