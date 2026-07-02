import unittest

from rag_pipeline.transform.cleaner import WikipediaArticleCleaner


class CleanerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cleaner = WikipediaArticleCleaner()

    def test_removes_standard_markup(self) -> None:
        text = (
            "__NOTOC__\n"
            "{{Infobox}}\n"
            "Tiếng Việt là ngôn ngữ.\n"
            "<ref>tham khảo</ref>\n"
            "[[Thể loại:Ngôn ngữ]]\n"
            "Được sử dụng rộng rãi.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("__NOTOC__", cleaned)
        self.assertNotIn("{{Infobox}}", cleaned)
        self.assertNotIn("<ref>tham khảo</ref>", cleaned)
        self.assertIn("Tiếng Việt là ngôn ngữ.", cleaned)
        self.assertIn("Được sử dụng rộng rãi.", cleaned)

    def test_removes_pipe_lines_and_closing_braces(self) -> None:
        text = (
            "| purpose = Đảm bảo phát triển Internet.\n"
            "| region_served = Toàn cầu\n"
            "}}\n"
            "Internet Society là tổ chức quốc tế.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| purpose", cleaned)
        self.assertNotIn("| region_served", cleaned)
        self.assertNotIn("}}", cleaned)
        self.assertIn("Internet Society là tổ chức quốc tế.", cleaned)

    def test_unwraps_wikilinks(self) -> None:
        text = "Xem thêm [[Việt Nam|Việt Nam]] và [[Lịch sử Việt Nam]]."

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("[[", cleaned)
        self.assertNotIn("]]", cleaned)
        self.assertIn("Việt Nam", cleaned)
        self.assertIn("Lịch sử Việt Nam", cleaned)

    def test_strips_bold_italic(self) -> None:
        text = "Đây là '''chữ đậm''' và ''chữ nghiêng''."

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("'''", cleaned)
        self.assertNotIn("''", cleaned)
        self.assertIn("chữ đậm", cleaned)
        self.assertIn("chữ nghiêng", cleaned)

    def test_handles_infobox_before_prose(self) -> None:
        """Real-world case: entry starts with template artifacts then prose."""
        text = (
            "(Hà Nội)\n"
            "(Huế)\n"
            "~ (TP. Hồ Chí Minh)\n"
            "}}\n"
            "| speakers = L1: triệu\n"
            "| ethnicity = Việt (Kinh)\n"
            "\n"
            "Tiếng Việt hay tiếng Kinh là một ngôn ngữ.\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| speakers", cleaned)
        self.assertNotIn("| ethnicity", cleaned)
        self.assertNotIn("}}", cleaned)
        self.assertIn("Tiếng Việt hay tiếng Kinh là một ngôn ngữ.", cleaned)

    def test_entry_with_only_template_garbage_has_no_markup(self) -> None:
        """Trang_Chính-style: only infobox links, no real prose — markup removed."""
        text = (
            "__NOTOC____NOEDITSECTION__\n"
            "| links =\n"
            "Lưu trữ\n"
            "Thêm bài viết chọn lọc\n"
            "Ứng cử viên\n"
            "}}\n"
        )

        cleaned = self.cleaner.clean(text)

        self.assertNotIn("| links", cleaned)
        self.assertNotIn("__NOTOC__", cleaned)
        self.assertNotIn("}}", cleaned)

    def test_already_clean_text_is_unchanged(self) -> None:
        text = "World Wide Web Consortium (W3C) là tổ chức tiêu chuẩn quốc tế."

        cleaned = self.cleaner.clean(text)

        self.assertEqual(text, cleaned)


if __name__ == "__main__":
    unittest.main()
