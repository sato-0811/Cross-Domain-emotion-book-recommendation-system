from __future__ import annotations

import unittest

from src.book.make_book_embeddings import split_into_blocks


class SplitIntoBlocksTest(unittest.TestCase):
    def test_overlapping_windows_cover_text(self) -> None:
        text = "abcdef" * 400  # 2400 chars
        blocks = split_into_blocks(text, block_size=1000, overlap=100)

        self.assertGreaterEqual(len(blocks), 3)
        self.assertEqual(blocks[0].start_char, 0)
        self.assertEqual(blocks[0].end_char, 1000)
        self.assertEqual(blocks[1].start_char, 900)
        self.assertEqual(blocks[1].end_char, 1900)
        self.assertEqual(blocks[-1].end_char, len(text))
        self.assertTrue(all(block.valid for block in blocks))

    def test_empty_text_returns_single_invalid_block(self) -> None:
        blocks = split_into_blocks("", block_size=1000, overlap=100)
        self.assertEqual(len(blocks), 1)
        self.assertFalse(blocks[0].valid)
        self.assertEqual(blocks[0].text, "")


if __name__ == "__main__":
    unittest.main()
